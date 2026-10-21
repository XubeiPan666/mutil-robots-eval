# Local RoboFactory/MAS-VLA adaptation of ManiSkill/RoboCasa scene code.
# Modified for multi-agent tasks and local scene/recording integration.
# See THIRD_PARTY_NOTICES.md and licenses/ for upstream licenses.

from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import Fixture


class Hood(Fixture):
    """
    Hood fixture class
    """

    def __init__(
        self, xml="fixtures/hoods/basic_silver.xml", name="hood", *args, **kwargs
    ):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    @property
    def nat_lang(self):
        return "range hood"
