import numpy as np
import pandas as pd
from pathlib import Path

import h5py
import hdf5storage # conda install hdf5storage

from pyphoplacecellanalysis.PhoPositionalData.load_exported import import_mat_file


import scipy.io # used for not HDF format files:
from warnings import warn

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.GUI.PyQtPlot.Examples.pyqtplot_DataTreeWidget import plot_dataTreeWidget # for GUI

class MatFileBrowser(object):
    """ A helper class that allows the user to interactively browser .mat (MATLAB files in a filesystem directory) 
    
    NOTE: This actually does work (as a primise) but it displays the data very poorly. 
    
    """
    
    @classmethod
    def discover_mat_files(cls, basedir, recursive=True):
        """ By default it attempts to find the all *.mat files in the root of this basedir
        Example:
            basedir: Path(r'R:\data\Bapun\Day5TwoNovel')
            session_name: 'RatS-Day5TwoNovel-2020-12-04_07-55-09'
        """
        # Find the only .xml file to obtain the session name 
        if recursive:
            glob_pattern = "**/*.mat"
        else:
            glob_pattern = "*.mat"
        mat_files = sorted(basedir.glob(glob_pattern))
        return mat_files # 'RatS-Day5TwoNovel-2020-12-04_07-55-09'
    
    @classmethod
    def whos_mat_file(cls, mat_file, ignored_keys = ['#subsystem#', '#refs#'], debug_h5py_dump=False, debug_print=False):
        """ Prints the names of the variables within the .mat file without having to load them.
            Performs two methods: first HDF5-based for newer .mat files, and then falling back to scipy.io.whosmat based for older ones.
        
        Returns:
            loaded_data: dict<Path, List<str>> - each path has a value of the found top-level variable names for that file
        """
        meta_info_dict = {}
        
        try:
            with h5py.File(mat_file,'r') as f:
                found_data_dict = {a_key:meta_info_dict for a_key in list(f.keys()) if a_key not in ignored_keys} ## filter the ignored keys
                # found_keys = [a_key for a_key in list(f.keys()) if a_key not in ignored_keys] ## filter the ignored keys
                if debug_h5py_dump:
                    found_data_dict['__dump__'] = f
                    
        except OSError as e:
            # Try whosmat:
            try:
                found_temp_data_tuples = scipy.io.whosmat(mat_file) # [('behavioral_epochs', (3, 6), 'double'), ('behavioral_periods', (668, 6), 'double')]
                found_data_dict = {a_key:{'size': a_data_size, 'type': a_data_type} for a_key, a_data_size, a_data_type in found_temp_data_tuples if a_key not in ignored_keys}
                # found_keys = [a_key for a_key, a_data_size, a_data_type in found_temp_data_tuples if a_key not in ignored_keys] ## filter the ignored keys    
            except OSError as e:
                # Actually failed
                warn(f'{mat_file} - !!ERROR: {e}')
                found_data_dict = None
                
        if debug_print:
            print(f'{mat_file}: {found_data_dict}')
        return found_data_dict
    
    @classmethod
    def whos_mat_files(cls, found_files, ignored_keys = ['#subsystem#', '#refs#'], debug_print=False):
        """ Prints the names of the variables within the .mat files without having to load them.
            Performs two methods: first HDF5-based for newer .mat files, and then falling back to scipy.io.whosmat based for older ones.
        
        Returns:
            loaded_data: dict<Path, List<str>> - each path has a value of the found top-level variable names for that file
        """
        return {found_file:cls.whos_mat_file(found_file, ignored_keys=ignored_keys, debug_print=debug_print) for found_file in found_files}
        # loaded_data = {}
        # for found_file in found_files:
        #     loaded_data[found_file] = cls.whos_mat_file(found_file, debug_print=debug_print)
        # return loaded_data
    
    @classmethod
    def build_browsing_gui(cls, loaded_data_dict, debug_print=False):
        """

        Args:
            loaded_data_dict (_type_): _description_
            debug_print (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
   
        Usage:  
            mat_import_parent_path = Path(r'R:\data\RoyMaze1')
            found_files = MatFileBrowser.discover_mat_files(mat_import_parent_path, recursive=True)
            found_file_variables_dict = MatFileBrowser.whos_mat_files(found_files)
            tree, app = MatFileBrowser.build_browsing_gui(found_file_variables_dict)
            tree.show()
        """
        # d = {
        #     'loaded_data_dict':loaded_data_dict
        # }
        # d = {str(a_file):file_keys for a_file, file_keys in loaded_data_dict.items()} # as simple array
        # d = {str(a_file):{a_file_key:'TODO' for a_file_key in file_keys} for a_file, file_keys in loaded_data_dict.items()} # as nested variable names
        d = {str(a_file):file_info_dict for a_file, file_info_dict in loaded_data_dict.items()} # as nested variable names
        # basedir
        tree, app = plot_dataTreeWidget(data=d, title=f'MAT Browsing GUI: {len(loaded_data_dict)} .mat files')
        return tree, app
    


def main():
    wants_recurrsive_mat_file_search = False
    # mat_import_parent_path = Path(r'R:\data\RoyMaze1')
    mat_import_parent_path = Path(r'W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15')
    found_files = MatFileBrowser.discover_mat_files(mat_import_parent_path, recursive=wants_recurrsive_mat_file_search)
    found_file_variables_dict = MatFileBrowser.whos_mat_files(found_files)
    tree, app = MatFileBrowser.build_browsing_gui(found_file_variables_dict)
    tree.show()
    pg.exec()
    
    
if __name__ == "__main__":
    main()