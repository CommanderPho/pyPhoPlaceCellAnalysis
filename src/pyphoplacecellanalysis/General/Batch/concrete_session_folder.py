from enum import Enum, unique
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder
from neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat import KDibaOldDataSessionFormatRegisteredClass  # noqa: F401 - registers format_name='kdiba'
from neuropy.utils.mixins.AttrsClassHelpers import custom_define, serialized_attribute_field
from neuropy.utils.result_context import IdentifyingContext
from pyphocorehelpers.Filesystem.path_helpers import build_cross_root_copydict, convert_filelist_to_new_parent, copy_movedict


@unique
class BackupMethods(Enum):
    CommonTargetDirectory = "COMMON_TARGET_DIR" # copies all files to the same output folder, meaning they need a prefix or suffix to identify their session added to their name
    RenameInSourceDirectory = "RENAME_IN_SOURCE_DIR" # copies to the same parent directory as the source file, but the copy has a prefix/suffix appended to the name


@custom_define(slots=False)
class ConcreteSessionFolder:
    """ a concrete representation of a session on disk """
    context: IdentifyingContext = serialized_attribute_field()
    path: Path = serialized_attribute_field()

    @property
    def session_pickle(self) -> Path:
        return self.path.joinpath('loadedSessPickle.pkl').resolve()

    @property
    def output_folder(self) -> Path:
        return self.path.joinpath('output').resolve()

    @property
    def pipeline_results_h5(self) -> Path:
        return self.output_folder.joinpath('pipeline_results.h5').resolve()

    @property
    def global_computation_result_pickle(self) -> Path:
        return self.output_folder.joinpath('global_computation_results.pkl').resolve()

    @classmethod
    def backup_output_files(cls, good_session_concrete_folders: List["ConcreteSessionFolder"], backup_mode: BackupMethods=BackupMethods.CommonTargetDirectory, target_dir: Optional[Path]=None, rename_backup_suffix: Optional[str]=None, skip_non_extant_src_files:bool=True, only_include_file_types=None, debug_print=False):
        """ builds the copydict and actually performs the copy

        """
        copy_dict = cls.backup_output_files(good_session_concrete_folders, backup_mode=backup_mode, target_dir=target_dir, rename_backup_suffix=rename_backup_suffix, skip_non_extant_src_files=skip_non_extant_src_files, only_include_file_types=only_include_file_types, debug_print=debug_print)
        moved_files_dict_files = copy_movedict(copy_dict)
        return moved_files_dict_files

    @classmethod
    def build_backup_copydict(cls, good_session_concrete_folders: List["ConcreteSessionFolder"], backup_mode: BackupMethods=BackupMethods.CommonTargetDirectory, target_dir: Optional[Path]=None, rename_backup_suffix: Optional[str]=None, rename_backup_basename_fn: Optional[Callable]=None, skip_non_extant_src_files:bool=True,
                               only_include_file_types=['local_pkl', 'global_pkl','h5'], custom_file_types_dict=None, debug_print=False):
        """ backs up the list of backup files to a specified target_dir.

        ## Usage 1:
            target_dir = Path('/home/halechr/cloud/turbo/Pho/Output/across_session_results/2023-10-03').resolve()
            copy_dict = ConcreteSessionFolder.build_backup_copydict(good_session_concrete_folders, target_dir=target_dir)
            copy_dict

        ## Usage 2:
            copy_dict = ConcreteSessionFolder.build_backup_copydict(good_session_concrete_folders, backup_mode=BackupMethods.RenameInSourceDirectory, rename_backup_suffix='2023-10-05')
            copy_dict


        Parameters:
            target_dir: only used if (backup_mode.name == BackupMethods.CommonTargetDirectory.name)
            rename_backup_suffix: Optional[str] only used if (backup_mode.name == BackupMethods.RenameInSourceDirectory.name)
            only_include_file_types: subet of file types to include: ['local_pkl', 'global_pkl','h5']


        """
        def _default_rename_basename_fn(session_context: Optional[IdentifyingContext], session_descr: Optional[str], basename: str, *args, separator_char: str = "_"):
            _filename_list = []
            if session_context is not None:
                session_descr = session_context.session_name # '2006-6-07_16-40-19'
            if session_descr is not None:
                _filename_list.append(session_descr)
            _filename_list.append(basename)
            if len(args) > 0:
                _filename_list.extend([str(a_part) for a_part in args if a_part is not None])
            return separator_char.join(_filename_list)


        if rename_backup_suffix is not None:
            assert (backup_mode.name == BackupMethods.RenameInSourceDirectory.name), f"rename_backup_suffix: {rename_backup_suffix} is only used if (backup_mode.name == BackupMethods.RenameInSourceDirectory.name), but backup_mode: {backup_mode} and rename_backup_suffix is not None!"
        if backup_mode.name == BackupMethods.RenameInSourceDirectory.name:
            assert rename_backup_suffix is not None, f"rename_backup_suffix is required if backup_mode == BackupMethods.RenameInSourceDirectory"

        if target_dir is not None:
            assert (backup_mode.name == BackupMethods.CommonTargetDirectory.name)
        if backup_mode.name == BackupMethods.CommonTargetDirectory.name:
            assert target_dir is not None
            target_dir.mkdir(parents=True, exist_ok=True)

        copy_dict = {}

        for a_session_folder in good_session_concrete_folders:
            session_descr: str = a_session_folder.context.get_description()
            if debug_print:
                print(f'a_session_folder: {session_descr}')
            src_files_dict = {'h5':a_session_folder.pipeline_results_h5, 'local_pkl':a_session_folder.session_pickle, 'global_pkl':a_session_folder.global_computation_result_pickle}
            if custom_file_types_dict is not None:
                ## add the custom filetypes if needed
                if only_include_file_types is None:
                    only_include_file_types = [] # empty type, we'll add the custom ones

                for k, v in custom_file_types_dict.items():
                    src_files_dict[k] = v(a_session_folder)
                    only_include_file_types.append(k) ## add the custom filetype to be included

            for src_file_kind, src_file in src_files_dict.items():
                if src_file_kind in (only_include_file_types or ['local_pkl', 'global_pkl','h5']):
                    if debug_print:
                        print(f'a_session_folder.src_file: {src_file}')
                    if skip_non_extant_src_files and (src_file is None) or (not src_file.exists()):
                        if debug_print:
                            print(f'src_file: "{src_file}" does not exist and skip_non_extant_src_files==True, so omitting from output copy_dict')
                    else:
                        # src_file: Path = a_session_folder.pipeline_results_h5
                        basename: str = src_file.stem
                        if backup_mode.name == BackupMethods.CommonTargetDirectory.name:
                            if rename_backup_basename_fn is not None:
                                final_dest_basename:str = rename_backup_basename_fn(a_session_folder.context, session_descr, basename)
                            else:
                                final_dest_basename:str = '_'.join([session_descr, basename])

                            final_dest_name:str = f'{final_dest_basename}{src_file.suffix}'
                            if debug_print:
                                print(f'\tfinal_dest_name: {final_dest_name}')
                            dest_path: Path = target_dir.joinpath(final_dest_name).resolve()
                        elif backup_mode.name == BackupMethods.RenameInSourceDirectory.name:
                            assert rename_backup_suffix is not None
                            target_dir = src_file.parent
                            if rename_backup_basename_fn is not None:
                                final_dest_basename:str = rename_backup_basename_fn(None, None, basename, rename_backup_suffix)
                            else:
                                final_dest_basename:str = '_'.join([basename, rename_backup_suffix])

                            final_dest_name:str = f'{final_dest_basename}{src_file.suffix}'
                            if debug_print:
                                print(f'\tfinal_dest_name: {final_dest_name}')
                            dest_path: Path = target_dir.joinpath(final_dest_name).resolve()
                        else:
                            raise ValueError

                        copy_dict[src_file] = dest_path
        return copy_dict


    def _get_session_stem_from_xml(self) -> Optional[str]:
        xml_files = list(self.path.glob('*.xml'))
        if len(xml_files) == 0:
            return None
        return xml_files[0].stem


    def discover_syncable_result_files(self, include_preprocessing_npy: bool = True, extra_globs: Optional[List[str]] = None) -> List[Path]:
        """Return extant paths for core pipeline outputs and optional session-preprocessing .npy files in the session folder.

        Usage:
            from pyphoplacecellanalysis.General.Batch.concrete_session_folder import ConcreteSessionFolder
            syncable_files = a_session_folder.discover_syncable_result_files()
        """
        syncable_files: List[Path] = []
        core_pipeline_files = [self.session_pickle, self.global_computation_result_pickle, self.pipeline_results_h5]
        for a_file in core_pipeline_files:
            if a_file.is_file():
                syncable_files.append(a_file.resolve())
        if include_preprocessing_npy:
            session_stem = self._get_session_stem_from_xml()
            if session_stem is not None:
                for a_npy_file in self.path.glob(f'{session_stem}*.npy'):
                    if a_npy_file.is_file():
                        syncable_files.append(a_npy_file.resolve())
        if extra_globs is not None:
            for a_glob_pattern in extra_globs:
                for a_matched_file in self.path.glob(a_glob_pattern):
                    if a_matched_file.is_file():
                        syncable_files.append(a_matched_file.resolve())
        return list(dict.fromkeys(syncable_files))


    @classmethod
    def build_cross_root_results_sync_copydict(cls, fast_session_folders: List["ConcreteSessionFolder"], session_fast_roots: Dict[IdentifyingContext, Path], archive_data_root: Path, include_preprocessing_npy: bool = True, extra_globs: Optional[List[str]] = None, skip_if_dest_newer_or_equal: bool = True, debug_print: bool = False) -> Dict[Path, Path]:
        """Build a fast->archive hierarchical copydict for computed session results, skipping sessions already on archive root or missing archive folders.

        Usage:
            from pyphoplacecellanalysis.General.Batch.concrete_session_folder import ConcreteSessionFolder
            copy_dict = ConcreteSessionFolder.build_cross_root_results_sync_copydict(good_session_concrete_folders, session_global_data_root_parent_paths, archive_data_root)
        """
        archive_data_root = Path(archive_data_root).resolve()
        sync_copydict: Dict[Path, Path] = {}
        for a_session_folder in fast_session_folders:
            fast_data_root = session_fast_roots.get(a_session_folder.context)
            if fast_data_root is None:
                if debug_print:
                    print(f'WARN: skipping {a_session_folder.context}: no fast data root in session_fast_roots')
                continue
            fast_data_root = Path(fast_data_root).resolve()
            if fast_data_root == archive_data_root:
                if debug_print:
                    print(f'skipping {a_session_folder.context}: fast root equals archive root ({fast_data_root})')
                continue
            archive_session_basedir = convert_filelist_to_new_parent([a_session_folder.path.resolve()], original_parent_path=fast_data_root, dest_parent_path=archive_data_root)[0].resolve()
            if not archive_session_basedir.is_dir():
                print(f'WARN: skipping {a_session_folder.context}: archive session folder missing: {archive_session_basedir}')
                continue
            source_files = a_session_folder.discover_syncable_result_files(include_preprocessing_npy=include_preprocessing_npy, extra_globs=extra_globs)
            if len(source_files) == 0:
                if debug_print:
                    print(f'skipping {a_session_folder.context}: no syncable source files found under {a_session_folder.path}')
                continue
            session_copydict = build_cross_root_copydict(source_files, source_data_root=fast_data_root, dest_data_root=archive_data_root, skip_if_dest_newer_or_equal=skip_if_dest_newer_or_equal)
            if debug_print:
                print(f'{a_session_folder.context}: {len(session_copydict)} file(s) to sync from {fast_data_root} -> {archive_data_root}')
            sync_copydict.update(session_copydict)
        return sync_copydict


    @classmethod
    def build_concrete_session_folders(cls, global_data_root_parent_path: Path, included_session_contexts: list, debug_print=False) -> List["ConcreteSessionFolder"]:
        """

        good_session_concrete_folders = ConcreteSessionFolder.build_concrete_session_folders(global_data_root_parent_path, included_session_contexts)

        """
        assert global_data_root_parent_path.exists(), f"global_data_root_parent_path: {global_data_root_parent_path} does not exist! Is the right computer's config commented out above?"

        known_data_session_type_properties_dict = DataSessionFormatRegistryHolder.get_registry_known_data_session_type_dict()
        active_data_session_types_registered_classes_dict = DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()

        all_data_mode_names = [a_ctxt.format_name for a_ctxt in included_session_contexts] # ['kdiba', ...]
        active_data_mode_name: str = all_data_mode_names.pop(0) # 'kdiba'
        assert np.all([(v == active_data_mode_name) for v in all_data_mode_names]), f"all contexts must be from the same data mode (arbitrarily). active_data_mode_name: {active_data_mode_name}, all_data_mode_names: {all_data_mode_names}"

        ## Get known properties for this type:
        active_data_mode_registered_class = active_data_session_types_registered_classes_dict[active_data_mode_name]
        active_data_mode_type_properties = known_data_session_type_properties_dict[active_data_mode_name]

        ## get specifics using the known properties:
        output_session_basedir_dict = active_data_mode_registered_class.build_session_basedirs_dict(global_data_root_parent_path, debug_print=debug_print)
        included_output_session_basedir_dict = {a_context:a_basedir for a_context, a_basedir in output_session_basedir_dict.items() if a_context in included_session_contexts}

        good_session_concrete_folders = [ConcreteSessionFolder(a_context, a_basedir) for a_context, a_basedir in included_output_session_basedir_dict.items()]
        return good_session_concrete_folders
