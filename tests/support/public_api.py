"""The public API: what the public packages export in ``__all__``, and the public methods and properties of the classes
among them (``forward`` aside: torch's hook, not a user call)."""
import importlib
import inspect

PACKAGES = ("alignair", "alignair.errors", "alignair.io", "alignair.registry", "alignair.reference",
            "alignair.model_file", "alignair.analyze", "alignair.evaluate")


def public_api():
    """``(qualified name, class or function)`` for every public class and function, and every public method or
    property of those classes."""
    for package_name in PACKAGES:
        package = importlib.import_module(package_name)
        for name in package.__all__:
            value = getattr(package, name)
            if inspect.isclass(value) or inspect.isfunction(value):
                yield f"{package_name}.{name}", value
                if inspect.isclass(value):
                    for attribute, member in vars(value).items():
                        if attribute.startswith("_") or attribute == "forward":
                            continue
                        function = member.fget if isinstance(member, property) else getattr(member, "__func__", member)
                        if inspect.isfunction(function):
                            yield f"{package_name}.{name}.{attribute}", function
