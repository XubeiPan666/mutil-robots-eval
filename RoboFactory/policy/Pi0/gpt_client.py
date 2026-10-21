import os
import base64
from io import BytesIO
import numpy as np
from PIL import Image
from openai import OpenAI
import time
from datetime import datetime

class GPTClient:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY before using the GPT scene planner.")
        self.api_base = os.getenv(
            "OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        )
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        self.model = "gpt-4.1" 
        
        # Create log directory if it doesn't exist
        self.log_dir = os.path.join(os.path.dirname(__file__), 'gpt_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create a new log file for each session
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.log_dir, f'gpt_responses_{timestamp}.log')

    def _encode_image_from_file(self, image_path):
        """Convert image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _encode_image(self, image_array):
        """Convert numpy array to base64 string."""
        image = Image.fromarray(image_array)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _log_interaction(self, prompt, response, actions):
        """Log the GPT interaction to file."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Prompt:\n{prompt}\n\n")
                f.write(f"Response:\n{response}\n")
                f.write(f"Selected Actions: {actions}\n")
                f.write(f"{'='*50}\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")

    def analyze_scene_from_files(self, image_file_paths, agent_states, task_prompt):
        """
        Analyze the scene using GPT-4 by reading images from files.
        
        Args:
            image_file_paths: Dict containing file paths for each agent and global view
            agent_states: Dict containing agent joint positions and gripper states
            task_prompt: Overall task description
            
        Returns:
            Dict containing instructions for each agent
        """
        # Predefined valid instructions for each phase
        phase_instructions = {
            1: {
                "agent0": "Lift the object and place it on the target.",
                "agent1": "Lift the object and wait.",
                "agent2": "Lift the object and wait."
            },
            2: {
                "agent0": "Return and wait for the other robots.",
                "agent1": "Place the object on the target.", 
                "agent2": "Wait for the other robots."
            },
            3: {
                "agent0": "Wait for the other robots.",
                "agent1": "Return and wait for the other robots.",
                "agent2": "Place the object on the target."
            },
            4: {
                "agent0": "xxxxxx",
                "agent1": "xxxxxx",
                "agent2": "xxxxxx"
            }
        }

        # Prepare agent state descriptions
        state_descriptions = []
        for agent_id, state in agent_states.items():
            gripper_state = "closed" if np.mean(state['gripper']) < 0.02 else "open"
            state_descriptions.append(f"{agent_id} gripper is {gripper_state}")

        # Build the content list with text and images
        content = [
            {
                "type": "input_text",
                "text": """Task: Stack three colored cubes in sequence to bullseye. First blue, second green, third red.

Scene Description:
- agent0 controls the robot arm near the BLUE cube
- agent1 controls the robot arm near the GREEN cube
- agent2 controls the robot arm near the RED cube

Phase Description:
Phase 1: All three agents first lift their respective cubes. This phase continues until agent0 places its cube onto the white-red target placed on the table and then **releases it with the gripper open**. then At this point, Phase 1 ends and Phase 2 begins.

Phase 2: This phase starts after Phase 1. It continues until agent1 places its cube on top of the cube placed by agent0, and then **releases it with the gripper open**. Once this is done, Phase 2 ends and Phase 3 begins.

Phase 3: This phase starts after Phase 2. It continues until agent2 places its cube on top of the stack (the cube placed by agent1), and then **releases it with the gripper open**. When this action is completed, Phase 3 ends.


Available actions for each agent:
1. "Lift the object and place it on the target."
2. "Lift the object and wait."
3. "Return and wait for the other robots."
4. "Place the object on the target."
5. "Wait for the other robots."

Please analyze the scene and provide:
1. Reasoning: Describe the current state of each cube and agent, and explain why certain actions are needed.
2. Actions: Based on your reasoning, select the appropriate action for each agent.


current phase instruction pairs are
        phase_instructions = {
            1: {
                "agent0": "Lift the object and place it on the target.",
                "agent1": "Lift the object and wait.",
                "agent2": "Lift the object and wait."
            },
            2: {
                "agent0": "Return and wait for the other robots.",
                "agent1": "Place the object on the target.", 
                "agent2": "Wait for the other robots."
            },
            3: {
                "agent0": "Wait for the other robots.",
                "agent1": "Return and wait for the other robots.",
                "agent2": "Place the object on the target."
            }
            }

Your response MUST be in the format:
{
    "reasoning": "Detailed explanation of the current scene and why each action is chosen",
    "actions": {
        "agent0": "<selected action>",
        "agent1": "<selected action>",
        "agent2": "<selected action>"
    }
}
"""
            }
        ]

        # Add images to content from files
        for agent_id, filepath in image_file_paths.items():
            if agent_id == "globalvllm":
                image_base64 = self._encode_image_from_file(filepath)
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_base64}"
                })
                continue

        # if 'global_view' in image_file_paths:
        #     image_base64 = self._encode_image_from_file(image_file_paths['global_view'])
        #     content.append({
        #         "type": "input_image",
        #         "image_url": f"data:image/jpeg;base64,{image_base64}"
        #     })

        # Build the input message
        input_message = [{
            "role": "user",
            "content": content
        }]

        try:
            response = self.client.responses.create(
                model=self.model,
                input=input_message
            )
            
            try:
                # Extract actions from response
                content = response.output_text.strip()
                
                # Validate that each action exists in one of the phase_instructions
                valid_actions = set()
                for phase in phase_instructions.values():
                    valid_actions.update(phase.values())
                
                # Parse the response to get actions
                try:
                    import json
                    response_dict = json.loads(content)
                    
                    # Extract actions from the nested structure
                    actions = response_dict.get('actions', {})
                    reasoning = response_dict.get('reasoning', '')
                    
                    # Validate actions
                    for agent, action in actions.items():
                        if action not in valid_actions:
                            print(f"Invalid action for {agent}: {action}, using default")
                            actions[agent] = phase_instructions[1][agent]
                    
                    # Log the interaction with reasoning
                    self._log_interaction(str(input_message), f"Reasoning: {reasoning}\nActions: {actions}", actions)
                    
                    return actions
                    
                except json.JSONDecodeError:
                    print("Could not parse JSON response, using default actions")
                    self._log_interaction(str(input_message), content, phase_instructions[1])
                    return phase_instructions[-1]
                
            except Exception as e:
                print(f"Error parsing GPT response: {e}")
                self._log_interaction(str(input_message), f"Error: {str(e)}", phase_instructions[1])
                return phase_instructions[-1]
                
        except Exception as e:
            print(f"Error in GPT API call: {e}")
            self._log_interaction(str(input_message), f"API Error: {str(e)}", phase_instructions[1])
            return phase_instructions[-1]

    def analyze_scene(self, images, agent_states, task_prompt):
        """
        Original analyze_scene method that works with numpy arrays.
        Kept for backward compatibility.
        """
        # Predefined valid instructions for each phase
        phase_instructions = {
            1: {
                "agent0": "Lift the object and place it on the target.",
                "agent1": "Lift the object and wait.",
                "agent2": "Lift the object and wait."
            },
            2: {
                "agent0": "Return and wait for the other robots.",
                "agent1": "Place the object on the target.", 
                "agent2": "Wait for the other robots."
            },
            3: {
                "agent0": "Wait for the other robots.",
                "agent1": "Return and wait for the other robots.",
                "agent2": "Place the object on the target."
            },
            4: {
                "agent0": "xxxxxx",
                "agent1": "xxxxxx",
                "agent2": "xxxxxx"
            }
        }

        # Prepare agent state descriptions
        state_descriptions = []
        for agent_id, state in agent_states.items():
            gripper_state = "closed" if np.mean(state['gripper']) < 0.02 else "open"
            state_descriptions.append(f"{agent_id} gripper is {gripper_state}")

        # Build the content list with text and images
        content = [
            {
                "type": "input_text",
                "text": """Task: Stack three colored cubes in sequence to bullseye. First blue, second green, third red.

Scene Description:
- agent0 controls the robot arm near the BLUE cube
- agent1 controls the robot arm near the GREEN cube
- agent2 controls the robot arm near the RED cube

Phase Description:
Phase 1: All three agents first lift their respective cubes. This phase continues until agent0 places its cube onto the bullseye and then releases it with the gripper. At this point, Phase 1 ends and Phase 2 begins.

Phase 2: This phase starts after Phase 1. It continues until agent1 places its cube on top of the cube placed by agent0, and then releases it with the gripper. Once this is done, Phase 2 ends and Phase 3 begins.

Phase 3: This phase starts after Phase 2. It continues until agent2 places its cube on top of the stack (the cube placed by agent1), and then releases it with the gripper. When this action is completed, Phase 3 ends.

Available actions for each agent:
1. "Lift the object and place it on the target."
2. "Lift the object and wait."
3. "Return and wait for the other robots."
4. "Place the object on the target."
5. "Wait for the other robots."

Please analyze the scene and provide:
1. Reasoning: Describe the current state of each cube and agent, and explain why certain actions are needed.
2. Actions: Based on your reasoning, select the appropriate action for each agent.

Your response MUST be in the format:
{
    "reasoning": "Detailed explanation of the current scene and why each action is chosen",
    "actions": {
        "agent0": "<selected action>",
        "agent1": "<selected action>",
        "agent2": "<selected action>"
    }
}

current phase instruction pairs are
        phase_instructions = {
            1: {
                "agent0": "Lift the object and place it on the target.",
                "agent1": "Lift the object and wait.",
                "agent2": "Lift the object and wait."
            },
            2: {
                "agent0": "Return and wait for the other robots.",
                "agent1": "Place the object on the target.", 
                "agent2": "Wait for the other robots."
            },
            3: {
                "agent0": "Wait for the other robots.",
                "agent1": "Return and wait for the other robots.",
                "agent2": "Place the object on the target."
            }
            }

You should refer this.


注意如果蓝色cube放上bullseye去了，agent1夹住松开了都满足了，就直接是phase2了，就是叠方块吗，一个叠一个，好好观察思考啊


Your response MUST be in the format:
{
    "reasoning": "Detailed explanation of the current scene and why each action is chosen",
    "actions": {
        "agent0": "<selected action>",
        "agent1": "<selected action>",
        "agent2": "<selected action>"
    }
}
"""
            }
        ]

        # Add images to content
        for agent_id, img in images.items():
            if agent_id != 'global_view':
                image_base64 = self._encode_image(img)
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_base64}"
                })

        if 'global_view' in images:
            image_base64 = self._encode_image(images['global_view'])
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{image_base64}"
            })

        # Build the input message
        input_message = [{
            "role": "user",
            "content": content
        }]

        try:
            response = self.client.responses.create(
                model=self.model,
                input=input_message
            )
            
            try:
                # Extract actions from response
                content = response.output_text.strip()
                
                # Validate that each action exists in one of the phase_instructions
                valid_actions = set()
                for phase in phase_instructions.values():
                    valid_actions.update(phase.values())
                
                # Parse the response to get actions
                try:
                    import json
                    response_dict = json.loads(content)
                    
                    # Extract actions from the nested structure
                    actions = response_dict.get('actions', {})
                    reasoning = response_dict.get('reasoning', '')
                    
                    # Validate actions
                    for agent, action in actions.items():
                        if action not in valid_actions:
                            print(f"Invalid action for {agent}: {action}, using default")
                            actions[agent] = phase_instructions[1][agent]
                    
                    # Log the interaction with reasoning
                    self._log_interaction(str(input_message), f"Reasoning: {reasoning}\nActions: {actions}", actions)
                    
                    return actions
                    
                except json.JSONDecodeError:
                    print("Could not parse JSON response, using default actions")
                    self._log_interaction(str(input_message), content, phase_instructions[1])
                    return phase_instructions[-1]
                
            except Exception as e:
                print(f"Error parsing GPT response: {e}")
                self._log_interaction(str(input_message), f"Error: {str(e)}", phase_instructions[1])
                return phase_instructions[-1]
                
        except Exception as e:
            print(f"Error in GPT API call: {e}")
            self._log_interaction(str(input_message), f"API Error: {str(e)}", phase_instructions[1])
            return phase_instructions[-1]

    def _get_default_instructions(self):
        """Return default instructions (phase 1) in case of error."""
        return {
            "agent0": "Lift the object and place it on the target.",
            "agent1": "Lift the object and wait.",
            "agent2": "Lift the object and wait."
        } 