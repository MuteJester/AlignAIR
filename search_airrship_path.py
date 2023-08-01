import site

# Get the site packages directories
site_packages = site.getsitepackages()

# Search for the airrship package in the site packages
airrship_path = None
for package_dir in site_packages:
    package_path = package_dir + '/airrship'
    try:
        __import__('airrship')
        airrship_path = package_path
        break
    except ImportError:
        continue

# Print the path to the airrship package
if airrship_path:
    print("airrship package is installed at:", airrship_path)
else:
    print("airrship package is not found in site packages.")