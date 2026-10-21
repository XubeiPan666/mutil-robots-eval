# Local RoboFactory/MAS-VLA adaptation of ManiSkill/RoboCasa scene code.
# Modified for multi-agent tasks and local scene/recording integration.
# See THIRD_PARTY_NOTICES.md and licenses/ for upstream licenses.

from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import Fixture


class Dishwasher(Fixture):
    """
    Dishwasher fixture class
    """

    def __init__(
        self,
        scene,
        xml="fixtures/appliances/dishwashers/pack_1/model.xml",
        name="dishwasher",
        *args,
        **kwargs
    ):
        super().__init__(
            scene, xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    @property
    def nat_lang(self):
        return "dishwasher"
