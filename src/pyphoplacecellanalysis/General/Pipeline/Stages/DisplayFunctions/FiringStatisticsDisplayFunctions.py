import numpy as np
try:
    import modin.pandas as pd # modin is a drop-in replacement for pandas that uses multiple cores
except ImportError:
    import pandas as pd # fallback to pandas when modin isn't available

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder
import pyphoplacecellanalysis.External.pyqtgraph as pg


class FiringStatisticsDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ DOCTODO """
    
    def display_firing_rate_trends(computation_result, active_config, enable_saving_to_disk=False, debug_print=False, **kwargs):
        """ DOCTODO
        
        TODO: this sucks, it displays a beeswarm plot that locks everything up for minutes
        
        
        Usage:
        
            # np.shape(active_one_step_decoder.active_time_windows) # (2892, 2)

            active_firing_rate_trends = computation_result.computed_data['firing_rate_trends']

            active_rolling_window_times = active_firing_rate_trends['active_rolling_window_times']
            mean_firing_rates = active_firing_rate_trends['mean_firing_rates']
            moving_mean_firing_rates_df = active_firing_rate_trends['moving_mean_firing_rates_df']
            moving_mean_firing_rates_df # 3969 rows x 43 columns

            # mean_firing_rates
            # pg.plot(mean_firing_rates)

            np.shape(moving_mean_firing_rates_df) # (3969, 43)
            good_only_moving_mean_firing_rates_df = moving_mean_firing_rates_df.dropna() # 3910 rows x 43 columns
            good_only_moving_mean_firing_rates_df.T
            err, win = _display_firing_rate_trends(good_only_moving_mean_firing_rates_df.T)
            win.show()

            # active_rolling_window_times # dtype='timedelta64[ns]', name='time_delta_sec', length=2900, freq='S'
            # pg.plot(moving_mean_firing_rates_df)

        """
        
        """ Computes DOC_TODO
        
        Requires:
            
        
        Provides:
            computation_result.computed_data['firing_rate_trends']
                ['firing_rate_trends']['active_rolling_window_times']
                ['firing_rate_trends']['mean_firing_rates']
                ['firing_rate_trends']['moving_mean_firing_rates_df']
        
        """
        def _display_firing_rate_trends(cell_firing_rate_samples, debug_print=False):
            """ a pyqtgraph-based plotting method """
            # Incoming data is (C,N): where C is the number of cells and N is the number of datapoints.
            num_cells = np.shape(cell_firing_rate_samples)[0]
            num_samples = np.shape(cell_firing_rate_samples)[1]
            assert (num_samples >= num_cells), f'num_samples should be greater than num_cells, but num_samples: {num_samples} and num_cells: {num_cells}! You probably meant the transpose of the data you passed in.'
            
            win = pg.plot()
            win.setWindowTitle('pyqtgraph beeswarm: Firing Rate Trends')

            if debug_print:
                print(f'np.shape(cell_firing_rate_samples): {np.shape(cell_firing_rate_samples)}, num_cells: {num_cells}, num_samples: {num_samples}')
            
            ## Make bar graph
            #bar = pg.BarGraphItem(x=range(4), height=data.mean(axis=1), width=0.5, brush=0.4)
            #win.addItem(bar)

            ## add scatter plots on top
            for i in np.arange(num_cells):
                curr_cell_samples = cell_firing_rate_samples.loc[i,:].to_numpy()
                if debug_print:
                    print(f'i: {i} - np.shape(curr_cell_samples): {np.shape(curr_cell_samples)}')
                xvals = pg.pseudoScatter(curr_cell_samples, spacing=0.4, bidir=True) * 0.2
                win.plot(x=xvals+i, y=curr_cell_samples, pen=None, symbol='o', symbolBrush=pg.intColor(i,6,maxValue=128))

            ## Make error bars
            plt_errorbar = pg.ErrorBarItem(x=np.arange(num_cells), y=cell_firing_rate_samples.mean(axis=1), height=cell_firing_rate_samples.std(axis=1), beam=0.5, pen={'color':'w', 'width':2})
            win.addItem(plt_errorbar)
            return plt_errorbar, win


        active_firing_rate_trends = computation_result.computed_data['firing_rate_trends']

        active_rolling_window_times = active_firing_rate_trends['active_rolling_window_times']
        mean_firing_rates = active_firing_rate_trends['mean_firing_rates']
        moving_mean_firing_rates_df = active_firing_rate_trends['moving_mean_firing_rates_df']
        # moving_mean_firing_rates_df # 3969 rows x 43 columns
        # mean_firing_rates
        # pg.plot(mean_firing_rates)
        # np.shape(moving_mean_firing_rates_df) # (3969, 43)
        good_only_moving_mean_firing_rates_df = moving_mean_firing_rates_df.dropna() # 3910 rows x 43 columns
        # good_only_moving_mean_firing_rates_df.T
        plt_errorbar, win = _display_firing_rate_trends(good_only_moving_mean_firing_rates_df.T)
        win.show()
        
        return plt_errorbar, win
        