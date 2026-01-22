import setuptools

setuptools.setup(name='holowan',
                 version='2909',
                 description='This is a Python SDK for HoloWAN_Huawei_edition',
                 author='chanyulin & kkchan',
                 packages=setuptools.find_packages(),
                 package_data={"holowan": ["resources/*.xml", "resources/*.ini"]}
                 )
