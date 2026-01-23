import os
import PySub
import shutil

init_location = os.path.realpath(PySub.__file__)
os.chdir(os.path.dirname(init_location))
os.chdir(r"..\Example_scripts")
print(os.getcwd())
py_files = [f for f in os.listdir(os.getcwd()) if f.endswith(".py")]
print(py_files)
for f in py_files:
    print(f)
    exec(open(f).read())
curdir = os.getcwd()
shutil.rmtree(os.path.join(curdir, "BucketEnsemble_test"))
