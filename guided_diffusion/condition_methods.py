from abc import ABC, abstractmethod
import torch

from .linalg import conjugate_gradient, gmres

__CONDITIONING_METHOD__ = {}


def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls

    return wrapper


def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser

    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == "gaussian":
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        elif self.noiser.__name__ == "poisson":
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement - Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError

        return norm_grad, norm

    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass


@register_conditioning_method(name="identity")
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t


@register_conditioning_method(name="projection")
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name="mcg")
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(
        self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs
    ):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(
            x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs
        )
        x_t -= norm_grad * self.scale

        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm


@register_conditioning_method(name="dps")
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(
            x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs
        )
        x_t -= norm_grad * self.scale
        return x_t, norm


@register_conditioning_method(name="dps+")
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get("num_sampling", 5)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling

        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm


@register_conditioning_method(name="pgdm")
class PseudoinverseGuided(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.maxiter = kwargs.get("maxiter", 1)

    @torch.no_grad()
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        x_s, x_t, y = x_t, x_prev, measurement

        alpha_s = kwargs["alpha"]
        alpha_t = kwargs["alpha_prev"]

        def A(x):
            return self.operator.forward(x, **kwargs)

        with torch.enable_grad():
            y_hat = A(x_0_hat)

        def At(y):
            return torch.autograd.grad(y_hat, x_0_hat, y, retain_graph=True)[0]

        var_x_xt = 1 - alpha_t
        var_y = self.noiser.sigma**2

        def cov_y_xt(v):
            return var_y * v + A(var_x_xt * At(v))

        error = y - y_hat
        grad = gmres(
            A=cov_y_xt,
            b=error,
            maxiter=self.maxiter,
        )
        grad = At(grad)
        grad = torch.autograd.grad(x_0_hat, x_t, grad)[0]
        grad = var_x_xt * grad

        x_s = x_s + torch.sqrt(alpha_t) * grad

        return x_s, torch.linalg.vector_norm(error)


@register_conditioning_method(name="mmps")
class MomentMatching(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.maxiter = kwargs.get("maxiter", 1)

    @torch.no_grad()
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        x_s, x_t, y = x_t, x_prev, measurement

        alpha_s = kwargs["alpha"]
        alpha_t = kwargs["alpha_prev"]
        scale = kwargs["scale"]

        def A(x):
            return self.operator.forward(x, **kwargs)

        with torch.enable_grad():
            y_hat = A(x_0_hat)

        def At(y):
            return torch.autograd.grad(y_hat, x_0_hat, y, retain_graph=True)[0]

        var_t = (1 - alpha_t) / torch.sqrt(alpha_t)
        var_y = self.noiser.sigma**2

        def cov_x_xt(v):
            return var_t * torch.autograd.grad(x_0_hat, x_t, v, retain_graph=True)[0]

        def cov_y_xt(v):
            return var_y * v + A(cov_x_xt(At(v)))

        error = y - y_hat
        grad = gmres(
            A=cov_y_xt,
            b=error,
            maxiter=self.maxiter,
        )
        grad = At(grad)
        grad = torch.autograd.grad(x_0_hat, x_t, grad)[0]
        grad = var_t * grad

        x_s = x_s + scale * grad

        return x_s, torch.linalg.vector_norm(error)
