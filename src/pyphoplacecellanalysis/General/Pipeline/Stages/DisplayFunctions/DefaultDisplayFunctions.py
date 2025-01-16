from neuropy.utils.mixins.dict_representable import overriding_dict_with # required for _display_2d_placefield_result_plot_raw
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphoplacecellanalysis.PhoPositionalData.plotting.placefield import plot_1d_placecell_validations
from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots


class DefaultDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    
    @function_attributes(short_name=None, tags=['1d_placefield_validations'], input_requires=["computation_result.computed_data['pf1D']"],
                          output_provides=[], uses=[], used_by=[], creation_date='2023-01-01 00:00', related_items=[])
    def _display_1d_placefield_validations(computation_result, active_config, **kwargs):
        """ Renders all of the flat 1D place cell validations with the yellow lines that trace across to their horizontally drawn placefield (rendered on the right of the plot) """
        # return out_figures_list
        return plot_1d_placecell_validations(computation_result.computed_data['pf1D'], active_config.plotting_config, **overriding_dict_with(lhs_dict={'modifier_string': 'lap_only', 'should_save': False}, **kwargs))


    @function_attributes(short_name=None, tags=['2d_placefield_result_plot_raw'], input_requires=["computation_result.computed_data['pf2D']"],
                          output_provides=[], uses=[], used_by=[], creation_date='2023-01-01 00:00', related_items=[])
    def _display_2d_placefield_result_plot_raw(computation_result, active_config, **kwargs):
        """ produces a stupid figure """
        fig = computation_result.computed_data['pf2D'].plot_raw(**overriding_dict_with(lhs_dict={'label_cells': True}, **kwargs)); # Plots an overview of each cell all in one figure
        # out_figures_list = [fig]
        # return out_figures_list
        return MatplotlibRenderPlots([fig])


# ==================================================================================================================== #
# Private Display Helpers                                                                                              #
# ==================================================================================================================== #
