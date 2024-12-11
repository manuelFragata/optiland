from abc import ABC


class RayAimer(ABC):

    def __init__(self, optic):
        raise NotImplementedError


class BisectionRayAimer(RayAimer):

    def __init__(self, optic):
        super().__init__(optic)

    def generate_rays(self, Hx, Hy, Px, Py, initial_rays):
        pass


class NewtonRayAimer(RayAimer):

    def __init__(self, optic):
        super().__init__(optic)

    def generate_rays(self, Hx, Hy, Px, Py, initial_rays):
        pass


class RayAimerFactory:

    @staticmethod
    def create(aiming_type, optic):
        if aiming_type == 'bisection':
            return BisectionRayAimer(optic)
        else:
            raise ValueError('Invalid aiming type: {}'.format(aiming_type))
