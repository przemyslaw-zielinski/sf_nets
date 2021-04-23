from .base import SimDataset

# dynamically import all subclasses
# of SimDataset in datasets package
from pathlib import Path
from inspect import isclass
from pkgutil import iter_modules
from importlib import import_module

package_dir = Path(__file__).resolve().parent
# iterate through the modules in the current package
for (_, module_name, _) in iter_modules([package_dir]):
    if module_name == 'base':
        continue

    # import the module and iterate through its attributes
    module = import_module(f"{__name__}.{module_name}")
    for attribute_name in dir(module):
        if attribute_name == 'SimDataset':
            continue
        attribute = getattr(module, attribute_name)

        if isclass(attribute) and issubclass(attribute, SimDataset):
            # Add the class to this package's variables
            globals()[attribute_name] = attribute
