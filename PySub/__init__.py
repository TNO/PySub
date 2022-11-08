import os
import sys

from pathlib import Path
for candidate in sys.path:
    if 'envs' in candidate:
        p = Path(candidate)
        environment_location = os.path.join(*p.parts[:p.parts.index('envs') + 2])
        break
    
os.environ['PROJ_LIB'] = os.path.join(environment_location, 'Library\share\proj\proj.db')
os.environ['GDAL_DATA'] = os.path.join(environment_location, 'Library\share')
os.environ['PROJ_DEBUG'] = '3'

import PySub.memory as memory
import PySub.SubsidenceModelGas as SubsidenceModelGas
import PySub.SubsidenceModelCavern as SubsidenceModelCavern
import PySub.SubsidenceSuite as SubsidenceSuite
import PySub.BucketEnsemble as BucketEnsemble
import PySub.Geometries as Geometries
import PySub.ProjectFolder as ProjectFolder
import PySub.SubsidenceKernel as SubsidenceKernel
import PySub.HorizontalDisplacementKernel as HorizontalDisplacement
