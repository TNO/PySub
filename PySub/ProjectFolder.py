import os
import shutil
from PySub import utils as _utils
from PySub import memory as _memory

import matplotlib.pyplot as plt
class ProjectFolder(object):
    """Class for managing a project folder
    """
    def __init__(self, folder, subfolders = []):
        """Initialize the ProjectFolder object. Also checks of the folder exists
        and creates it (if possible) when it doesn't exist. Does the same for any 
        indicated subfolders.

        Parameters
        ----------
        folder : str
            Path to a folder.
        subfolders : list str, optional
            List of strings with folder names. Not paths!. The default is [], 
            indicating no subfolders will be created.

        Raises
        ------
        Exception
            with invalid input.

        """
        if isinstance(folder, str):
            _utils.make_folder(folder)
            self.project_folder = folder
            for subfolder in subfolders:
                if isinstance(subfolder, str):
                    self.make_folder(subfolder)
        elif folder is None:
            self.project_folder = folder
        else:
            raise Exception(f"Invalid input type: {type(folder)}. Use a string representing a path.")
        
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return str('Project folder at:\n' + str(self.project_folder))
        
    def write_to_input(self, file, rename = None):
        """Copy an exisiting file to the input folder in the project tree.

        Parameters
        ----------
        file : str
            Location of an exisiting file.
        rename : str, optional
            New file name. The default is None.

        Raises
        ------
        Exception
            If the file does not exist.

        """
        if self.project_folder is not None:
            if os.path.isfile(file):
                file_basename, ext = os.path.splitext(file)
                if rename is None:
                    file_name = os.path.basename(file_basename)
                else:
                    _, rename_ext = os.path.splitext(rename)
                    if rename_ext:
                        ext = rename_ext
                    file_name = rename
                shutil.copyfile(file, self.input_file(file_name + ext))
            else:
                raise Exception(f'Invalid file. Cannot copy.')
            
    def move(self, new_folder):
        """Copy all the files in the project tree to a new location
        
        Parameters
        ----------
        new_folder : path
            Path to a folder.

        """
        
        if os.path.isdir(self.project_folder):
            old_folder = self.project_folder
            self.project_folder = new_folder
            _memory._move_tree(old_folder, self.project_folder)
            
            
    def _get_file(self, name, folder):
        if os.path.isdir(os.path.dirname(name)):
            return name
        if not hasattr(self, f'{folder}_folder') and self.project_folder is not None:
            self.make_folder(folder)
        if self.project_folder is not None:
            return os.path.join(getattr(self, f'{folder}_folder'), name)
    
    def input_file(self, name):
        """Get the path for a file in the input folder, given the name of the file.

        Parameters
        ----------
        name : str
            Name of the input file. Can have an extension (i.e.: .jpg or .txt).
            THe file does not have to exist, which is useful for saving files in 
            this folder.

        Returns
        -------
        str
            Path to the file in the input folder of the project tree.

        """
        return self._get_file(name, 'input')
    
    def output_file(self, name):
        """Get the path for a file in the output folder, given the name of the file.

        Parameters
        ----------
        name : str
            Name of the output file. Can have an extension (i.e.: .jpg or .txt).
            THe file does not have to exist, which is useful for saving files in 
            this folder.

        Returns
        -------
        str
            Path to the file in the output folder of the project tree.

        """
        output_file = self._get_file(name, 'output')
        
        return output_file
    
    def save_file(self, name):
        """Get the path for a file in the save folder, given the name of the file.

        Parameters
        ----------
        name : str
            Name of the save file. Can have an extension (i.e.: .jpg or .txt).
            THe file does not have to exist, which is useful for saving files in 
            this folder.

        Returns
        -------
        str
            Path to the file in the save folder of the project tree.

        """
        return self._get_file(name, 'save')
        
    def savefig(self, name):
        """Save the current figure to the output file with a certain name.

        Parameters
        ----------
        name : str
            save name of the figure.

        """
        if self.project_folder is not None:
            file_name = self.output_file(name)
            add = 1
            while os.path.isfile(file_name + '.png'):
                file_name = self._get_file(f'{name}_{add :03d}', 'output')
                add+=1
            plt.savefig(file_name, dpi = 'figure', bbox_inches = 'tight')
        
    def make_folder(self, folder):
        """Create a folder with a certain name

        Parameters
        ----------
        folder : str
            Name of the folder. This folder will be created as a part of the 
            project file tree.

       
        """
        if self.project_folder is not None:
            folder_location = os.path.join(self.project_folder, folder)
            setattr(self, f'{folder}_folder', folder_location)
            _utils.make_folder(getattr(self, f'{folder}_folder'))

