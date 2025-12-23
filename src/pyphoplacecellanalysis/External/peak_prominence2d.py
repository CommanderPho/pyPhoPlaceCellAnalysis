'''Compute peak prominence on 2d array using contour method.

Compute topographic prominence on a 2d surface. See
    https://en.wikipedia.org/wiki/Topographic_prominence
for more details.

This module takes a surface in R3 defined by 2D X, Y and Z arrays,
and use enclosing contours to define local maxima. The prominence of a local
maximum (peak) is defined as the height of the peak's summit above the
lowest contour line encircling it but containing no higher summit.

Optionally, peaks with small prominence or area can be filtered out.

Many of these terms come from the study of actual mountain ranges.

Terminology:
"col": In geomorphology, a col is the lowest point on a mountain ridge between two peaks.
"key col": a property of a peak; the highest col surrounding the peak - a unique point on this contour line 
"parent peak": a property of a peak; some higher mountain, selected according to various criteria. 

Notes:
"Peaks with high prominence tend to be the highest points around and are likely to have extraordinary views."


Author: guangzhi XU (xugzhi1987@gmail.com; guangzhi.xu@outlook.com)
Update time: 2018-11-10 16:03:49.
'''


#--------Import modules-------------------------
import numpy as np
from matplotlib.transforms import Bbox
from matplotlib.path import Path
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def isClosed(xs,ys):
    if np.alltrue([np.allclose(xs[0],xs[-1]),\
        np.allclose(ys[0],ys[-1]),xs.ptp(),ys.ptp()]):
        return True
    else:
        return False

def isContClosed(contour):
    x=contour.vertices[:, 0]
    y=contour.vertices[:, 1]
    return isClosed(x,y)

def polygonArea(x,y):
    if not isClosed(x,y):
        # here is a minor issue: isclosed() on lat/lon can be closed,
        # but after projection, unclosed. Happens to spurious small
        # contours usually a triangle. just return 0.
        return 0
    area=np.sum(y[:-1]*np.diff(x)-x[:-1]*np.diff(y))
    return np.abs(0.5*area)

def contourArea(contour):
    '''Compute area of contour
    <contour>: matplotlib Path obj, contour.

    Return <result>: float, area enclosed by <contour>.
    NOTE that <contour> is not necessarily closed by isClosed() method,
    it won't be when a closed contour has holes in it (like a doughnut). In such
    cases, areas of holes are subtracted.
    '''

    segs=contour.to_polygons()
    if len(segs)>1:
        areas=[]
        for pp in segs:
            xii=pp[:,0]
            yii=pp[:,1]
            areaii=polygonArea(xii,yii)
            areas.append(areaii)
        areas.sort()
        result=areas[-1]-np.sum(areas[:-1])
    else:
        x=contour.vertices[:, 0]
        y=contour.vertices[:, 1]
        result=polygonArea(x,y)

    return result

def polygonGeoArea(lons,lats,method='basemap',projection='cea',bmap=None, verbose=True):

    #------Use basemap to project coordinates------
    if method=='basemap':
        if bmap is None:
            from mpl_toolkits.basemap import Basemap

            lat1=np.min(lats)
            lat2=np.max(lats)
            lat0=np.mean(lats)
            lon1=np.min(lons)
            lon2=np.max(lons)
            lon0=np.mean(lons)

            if projection=='cea':
                bmap=Basemap(projection=projection,\
                        llcrnrlat=lat1,llcrnrlon=lon1,\
                        urcrnrlat=lat2,urcrnrlon=lon2)
            elif projection=='aea':
                bmap=Basemap(projection=projection,\
                        lat_1=lat1,lat_2=lat2,lat_0=lat0,lon_0=lon0,
                        llcrnrlat=lat1,llcrnrlon=lon1,\
                        urcrnrlat=lat2,urcrnrlon=lon2)

        xs,ys=bmap(lons,lats)

    #------Use pyproj to project coordinates------
    elif method=='proj':
        from pyproj import Proj

        lat1=np.min(lats)
        lat2=np.max(lats)
        lat0=np.mean(lats)
        lon0=np.mean(lons)

        pa=Proj('+proj=aea +lat_1=%f +lat_2=%f +lat_0=%f +lon_0=%f'\
                %(lat1,lat2,lat0,lon0))
        xs,ys=pa(lons,lats)

    result=polygonArea(xs,ys)

    return result

def contourGeoArea(contour,bmap=None):
    '''Compute area enclosed by latitude/longitude contour.
    Result in m^2
    '''

    segs=contour.to_polygons()
    if len(segs)>1:
        areas=[]
        for pp in segs:
            xii=pp[:,0]
            yii=pp[:,1]
            areaii=polygonGeoArea(xii,yii,bmap=bmap)
            areas.append(areaii)
        areas.sort()
        result=areas[-1]-np.sum(areas[:-1])
    else:
        x=contour.vertices[:, 0]
        y=contour.vertices[:, 1]
        result=polygonGeoArea(x,y,bmap=bmap)

    return result


def getProminence(var, step, ybin_centers=None, xbin_centers=None, min_depth=None, include_edge=True, min_area=None, max_area=None, area_func=contourArea, centroid_num_to_center=5, allow_hole=True, max_hole_area=None, verbose=False):
    '''Find 2d prominences of peaks.

    <var>: 2D ndarray, data to find local maxima. Missings (nans) are masked.
    <step>: float, contour interval. Finer (smaller) interval gives better accuarcy.
    <ybin_centers>, <xbin_centers>: 1d array, y and x coordinates of <var>. If not given,
                    use int indices.
    <min_depth>: float, filter out peaks with prominence smaller than this.
    <include_edge>: bool, whether to include unclosed contours that touch
                    the edges of the data, useful to include incomplete
                    contours.
    <min_area>: float, minimal area of the contour of a peak's col. Peaks with
                its col contour area smaller than <min_area> are discarded.
                If None, don't filter by contour area. If latitude and
                longitude axes available, compute geographical area in km^2.
    <max_area>: float, maximal area of a contour. Contours larger than
                <max_area> are discarded. If latitude and
                longitude axes available, compute geographical area in km^2.
    <area_func>: function obj, a function that accepts x, y coordinates of a 
                 closed contour and computes the inclosing area. Default
                 to contourArea().
    <centroid_num_to_center>: int, number of the smallest contours in a peak
                              used to compute peak center.
    <allow_hole>: bool, whether to discard tidy holes in contour that could arise
                  from noise.
    <max_hole_area>: float, if <allow_hole> is True, tidy holes with area
                     smaller than this are discarded.

    Return <result>: dict, keys: ids of found peaks.
                     values: dict, storing info of a peak:
            'id'        : int, id of peak,
            'height'    : max of height level,
            'col_level' : height level at col,
            'prominence': prominence of peak,
            'area'      : float, area of col contour. If latitude and 
                          longitude axes available, geographical area in
                          km^2. Otherwise, area in unit^2, unit is the same
                          as x, y axes,
            'contours'  : list, contours of peak from heights level to col,
                          each being a matplotlib Path obj
            'parent'    : int, id of a peak's parent. Heightest peak as a
                          parent id of 0.

    Author: guangzhi XU (xugzhi1987@gmail.com; guangzhi.xu@outlook.com)
    Update time: 2018-11-11 18:42:04.
    '''

    fig,ax=plt.subplots()

    def checkIn(cont1,cont2,lon1,lon2,lat1,lat2):
        fails=[]
        vs2=cont2.vertices
        for ii in range(len(vs2)):
            if not cont1.contains_point(vs2[ii]) and\
                not np.isclose(vs2[ii][0],lon1) and\
                not np.isclose(vs2[ii][0],lon2) and\
                not np.isclose(vs2[ii][1],lat1) and\
                not np.isclose(vs2[ii][1],lat2):
                fails.append(vs2[ii])
            if len(fails)>0:
                break
        return fails

    var=np.ma.masked_where(np.isnan(var),var).astype('float')
    needslerpx=True
    needslerpy=True
    if ybin_centers is None:
        ybin_centers=np.arange(var.shape[0])
        needslerpy=False
    if xbin_centers is None:
        xbin_centers=np.arange(var.shape[1])
        needslerpx=False

    if area_func==contourGeoArea:
        from mpl_toolkits.basemap import Basemap
        lat1=np.min(ybin_centers)
        lat2=np.max(ybin_centers)
        lon1=np.min(xbin_centers)
        lon2=np.max(xbin_centers)

        bmap=Basemap(projection='cea',\
                llcrnrlat=lat1,llcrnrlon=lon1,\
                urcrnrlat=lat2,urcrnrlon=lon2)

    vmax=np.nanmax(var)
    vmin=np.nanmin(var)
    step=abs(step)
    levels=np.arange(vmin,vmax+step,step).astype('float')

    npeak=0
    peaks={}
    prominence={}
    parents={}

    #----------------Get bounding box----------------
    #bbox=Bbox.from_bounds(xbin_centers[0],ybin_centers[0],np.ptp(xbin_centers),np.ptp(height))
    bbox=Path([[xbin_centers[0],ybin_centers[0]], [xbin_centers[0],ybin_centers[-1]],
        [xbin_centers[-1],ybin_centers[-1]], [xbin_centers[-1],ybin_centers[0]], [xbin_centers[0], ybin_centers[0]]])

    #If not allow unclosed contours, get all contours in one go
    if not include_edge:
        conts=ax.contour(xbin_centers,ybin_centers,var,levels)
        contours=conts.collections[::-1]
        got_levels=conts.cvalues
        if not np.all(got_levels==levels):
            levels=got_levels
        ax.cla()

    large_conts=[]

    #---------------Loop through levels---------------
    for ii,levii in enumerate(levels[::-1]):
        if verbose:
            print('# <getProminence>: Finding contour %f' %levii)

        #-Get a 2-level contour if allow unclosed contours-
        if include_edge:
            csii=ax.contourf(xbin_centers,ybin_centers,var,[levii,vmax+step]) ## Heavy-lifting code here. levii is the level
            csii=csii.collections[0]
            ax.cla()
        else:
            csii=contours[ii]

        #--------------Loop through contours at level--------------
        for jj, contjj in enumerate(csii.get_paths()):

            contjj.level=levii
            #contjj.is_edge=contjj.intersects_bbox(bbox,False) # False significant
            # this might be another matplotlib bug, intersects_bbox() used
            # to work
            contjj.is_edge=contjj.intersects_path(bbox, False) # False significant

            # NOTE: contjj.is_edge==True is NOT equivalent to
            # isContClosed(contjj)==False, unclosed contours inside boundaries
            # can happen when missings are present

            if not include_edge and contjj.is_edge:
                continue

            if not include_edge and not isContClosed(contjj):
                # Sometimes contours are not closed
                # even if not touching edge, this happens when missings
                # are present. In such cases, need to close it before
                # computing area. But even so, unclosed contours won't
                # contain any other, so might well just skip it.
                # the contourf() approach seems to be more robust in such 
                # cases.
                continue

            #--------------------Check area--------------------
            # if contour contains a big contour, skip area computation
            area_big=False
            for cii in large_conts:
                if contjj.contains_path(cii):
                    area_big=True
                    break

            if area_big:
                continue

            if area_func==contourGeoArea:
                contjj.area=area_func(contjj,bmap=bmap)/1e6
            else:
                contjj.area=area_func(contjj)

            if max_area is not None and contjj.area>max_area:
                large_conts.append(contjj)
                continue

            #----------------Remove small holes----------------
            segs=contjj.to_polygons()
            if len(segs)>1:
                contjj.has_holes=True
                if not allow_hole:
                    continue
                else:
                    if max_hole_area is not None:
                        areas=[]
                        if area_func==contourGeoArea:
                            areas=[polygonGeoArea(segkk[:,0],segkk[:,1],\
                                bmap=bmap)/1e6 for segkk in segs]
                        else:
                            areas=[polygonArea(segkk[:,0],segkk[:,1])\
                                    for segkk in segs]
                        areas.sort()
                        if areas[-2]>=max_hole_area:
                            continue

            else:
                contjj.has_holes=False

            if len(peaks)==0:
                npeak+=1
                peaks[npeak]=[contjj,]
                prominence[npeak]=levii
                parents[npeak]=0
            else:
                #-Check if new contour contains any previous ones-
                match_list=[]
                for kk,vv in peaks.items():
                    if contjj.contains_path(vv[-1]):
                        match_list.append(kk)
                    else:
                        # this is likely a bug in matplotlib. The contains_path()
                        # function is not entirely reliable when contours are
                        # touching the edge and step is small. Sometimes
                        # enclosing contours will fail the test. In such cases
                        # check all the points in cont2 with cont1.contains_point()
                        # if no more than 2 or 3 points failed, it is a pass.
                        # see https://stackoverflow.com/questions/47967359/matplotlib-contains-path-gives-unstable-results for more details.
                        # UPDATE: I've changed the method when 2
                        # contours compared are touching the edge: it seems that
                        # sometimes all points at the edge will fail so the
                        # failed number can go above 10 or even more. The new
                        # method compares the number of points that fail the contains_point()
                        # check with points at the edge. If all failing points are
                        # at the edge,report a contain relation
                        fail=checkIn(contjj,vv[-1],xbin_centers[0],xbin_centers[-1],ybin_centers[0], ybin_centers[-1])
                        if len(fail)==0:
                            match_list.append(kk)

                #---------Create new center if non-overlap---------
                if len(match_list)==0:
                    npeak+=1
                    peaks[npeak]=[contjj,]
                    prominence[npeak]=levii
                    parents[npeak]=0

                elif len(match_list)==1:
                    peaks[match_list[0]].append(contjj)

                else:
                    #------------------Filter by area------------------
                    if min_area is not None and len(match_list)>1:
                        match_list2=[]
                        for mm in match_list:
                            areamm=peaks[mm][-1].area
                            if areamm<min_area:
                                print (match_list)
                                print ('del by area',mm)
                                del peaks[mm]
                                del prominence[mm]
                                if mm in parents:
                                    del parents[mm]
                            else:
                                match_list2.append(mm)

                        match_list=match_list2

                    #------------------Get prominence------------------
                    if len(match_list)>1:
                        match_heights=[peaks[mm][0].level for mm in match_list]
                        max_idx=match_list[np.argmax(match_heights)]
                        for mm in match_list:
                            if prominence[mm]==peaks[mm][0].level and mm!=max_idx:
                                prominence[mm]=peaks[mm][0].level-levii
                                parents[mm]=max_idx
                        peaks[max_idx].append(contjj)

                    #---------------Filter by prominence---------------
                    if min_depth is not None and len(match_list)>1:
                        match_list2=[]
                        for mm in match_list:
                            if prominence[mm]<min_depth:
                                del peaks[mm]
                                del prominence[mm]
                                if mm in parents:
                                    del parents[mm]
                            else:
                                match_list2.append(mm)
                        match_list=match_list2

                    #-----------Add to all existing centers-----------    
                    #for mm in match_list:
                        #peaks[mm].append(contjj)

    # ==================================================================================================================== #
    #------------------Prepare output------------------
    result={}
    result_map=np.zeros(var.shape)
    parent_map=np.zeros(var.shape)-1
    id_map=np.zeros(var.shape)

    keys=list(peaks.keys())
    for ii in range(len(peaks)):
        kk=keys[ii]
        vv=peaks[kk]
        #--------------Remove singleton peaks--------------
        if len(vv)<2:
            continue
        
        lev_range=[cii.level for cii in vv]
        prokk=prominence[kk]

        #-------Use first few centroids to get center-------
        nc = min(centroid_num_to_center,len(vv))
        centerkk=np.array([jj.vertices.mean(axis=0) for jj in vv[:nc]])
        centerkk=np.mean(centerkk,axis=0)

        peakii={
            'id'         : kk,
            'height'  : np.max(lev_range),
            'col_level'  : np.min(lev_range),
            'prominence'  : prokk,
            'area'       : vv[-1].area,
            'contours'   : vv,
            'contour'    : vv[-1],
            'center'     : centerkk,
            'parent'     : parents[kk]
            }

        result[kk]=peakii
        # lerp1 (lienar interpolation) to get center indices
        if needslerpx:
            fitx=interp1d(xbin_centers,np.arange(var.shape[1]))
            xidx=fitx(centerkk[0])
        else:
            xidx=centerkk[0]

        if needslerpy:
            fity=interp1d(ybin_centers,np.arange(var.shape[0]))
            yidx=fity(centerkk[1])
        else:
            yidx=centerkk[1]

        xidx=np.around(xidx,0).astype('int')
        yidx=np.around(yidx,0).astype('int')

        id_map[yidx,xidx]=kk
        result_map[yidx,xidx]=prokk
        parent_map[yidx,xidx]=parents[kk]

    plt.close(fig)

    return result, id_map, result_map, parent_map


def compute_prominence_contours(xbin_centers, ybin_centers, slab, step=0.1, min_area=None, min_depth=0.2, include_edge=True, verbose=False, **kwargs):
    """ Simple wrapper around the getProminence function by Pho Hale
    xbin_centers and ybin_centers should be like *bin_labels not *bin
    slab should usually be transposed: tuning_curves[i].T
    
    Usage:        
        step = 0.2
        i = 0
        xx, yy, slab, peaks, idmap, promap, parentmap = perform_compute_prominence_contours(active_pf_2D_dt.xbin_labels, active_pf_2D_dt.ybin_labels, active_pf_2D.ratemap.tuning_curves[i].T, step=step)
        
        # Test plot the promenence result
        figure, (ax1, ax2, ax3, ax4) = plot_Prominence(xx, yy, slab, peaks, idmap, promap, parentmap, debug_print=False)

    """
    peaks_dict, id_map, prominence_map, parent_map = getProminence(slab, step, ybin_centers=ybin_centers, xbin_centers=xbin_centers, min_area=min_area, min_depth=min_depth, include_edge=include_edge, verbose=verbose, **kwargs)
    return xbin_centers, ybin_centers, slab, peaks_dict, id_map, prominence_map, parent_map


from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters


class PeakPromenence:
    """ 
        from pyphoplacecellanalysis.External.peak_prominence2d import PeakPromenence

    """


    @classmethod
    def _find_contours_at_levels(cls, xbin_centers, ybin_centers, slab, peak_probe_point, probe_levels):
        """ finds the contours containing the peak_probe_point at the specified probe_levels.
            performs slicing through desired z-values (1/2 prominence, etc) using contourf
            
            
        Inputs:
            peak_probe_point: a point (x, y) to use to validate or exclude found contours. This allows us to only get the contour the encloses a peak at a given level, not any others that may happen to be at that level as well.
            probe_levels: a list of z-values to slice at to find the contours
            
        Returns:
            a dict with keys of the probe_levels and values containing a list of their corresponding contours
        """
        vmax = np.nanmax(slab)
        fig, ax = plt.subplots()
        included_computed_contours = DynamicParameters.init_from_dict({}) 
        #---------------Loop through levels---------------
        for ii, levii in enumerate(probe_levels[::-1]):
            # Note that contourf requires at least 2 levels, hence the use of the vmax+1.0 term and accessing only the first item in the collection. Otherwise: "ValueError: Filled contours require at least 2 levels."
            csii = ax.contourf(xbin_centers, ybin_centers, slab, [levii, vmax+1.0]) ## Heavy-lifting code here. levii is the level
            csii = csii.collections[0]
            # ax.cla() ## TODO: this is the most computationally expensive part of the code, and it doesn't seem necissary
            #--------------Loop through contours at level--------------
            # find only the ones containing the peak_probe_point
            included_computed_contours[levii] = [contjj for jj, contjj in enumerate(csii.get_paths()) if contjj.contains_point(peak_probe_point)]
            n_contours = len(included_computed_contours[levii])
            assert n_contours <= 1, f"n_contours is supposed to be equal to be either 0 or 1 but len(included_computed_contours[levii]): {len(included_computed_contours[levii])}!"
            # assert n_contours == 1, f"contour_stats is supposed to be equal to 1 but len(included_computed_contours[levii]): {len(included_computed_contours[levii])}!"
            if n_contours == 0:
                warn( f"n_contours is 0 for level: {levii}")
                included_computed_contours[levii] = None # set to None
            else:                   
                included_computed_contours[levii] = included_computed_contours[levii][0] # unwrapped from the list format, it's just the single Path/Curve now
            
        plt.close(fig) # close the figure when done generating the contours to prevent an empty figure from showing
        return included_computed_contours

    @classmethod
    def _build_filtered_summits_analysis_results(cls, xbin, ybin, xbin_labels, ybin_labels, flat_peaks_df, active_eloy_analysis, slice_level_multiplier=0.5, minimum_included_peak_height=0.5, debug_print=False):
        """ builds the filtered summits analysis results dataframe and flat counts matrix 
        
        Usage:
            filtered_summits_analysis_df, pf_peak_counts_map = build_filtered_summits_analysis_results(active_pf_2D.xbin, active_pf_2D.ybin, active_pf_2D.xbin_labels, active_pf_2D.ybin_labels,
                                                                                            active_peak_prominence_2d_results, active_eloy_analysis, slice_level_multiplier=0.5, minimum_included_peak_height=1.0, debug_print = False)
                                                                                            
        """
        ## Find which position bin each peak falls in and add it to the flat_peaks_df:
        filtered_summits_analysis_df = flat_peaks_df[flat_peaks_df['peak_height'] >= minimum_included_peak_height].copy() # filter for peaks greater than 1.0Hz
        
        ## IMPORTANT: Filter by only one of the slice_levels before continuing, otherwise you're double-counting:
        filtered_summits_analysis_df = filtered_summits_analysis_df[filtered_summits_analysis_df['slice_level_multiplier'] == slice_level_multiplier].copy()

        ## Build outputs:
        n_xbins = len(xbin) - 1 # the -1 is to get the counts for the centers only
        n_ybins = len(ybin) - 1 # the -1 is to get the counts for the centers only
        pf_peak_counts_map = np.zeros((n_xbins, n_ybins), dtype=int) # create an initially zero matrix

        current_bin_counts = filtered_summits_analysis_df.value_counts(subset=['peak_center_binned_x', 'peak_center_binned_y'], normalize=False, sort=False, ascending=True, dropna=True) # current_bin_counts: a series with a MultiIndex index for each bin that has nonzero counts
        if debug_print:
            print(f'np.shape(current_bin_counts): {np.shape(current_bin_counts)}') # (247,)
        for (xbin_label, ybin_label), count in current_bin_counts.iteritems():
            if debug_print:
                print(f'xbin_label: {xbin_label}, ybin_label: {ybin_label}, count: {count}')
            try:
                pf_peak_counts_map[xbin_label-1, ybin_label-1] += count #if it's already a label, why are we subtracting 1?
            except IndexError as e:
                print(f'e: {e}\n filtered_summits_analysis_df: {np.shape(filtered_summits_analysis_df)}, current_bin_counts: {np.shape(current_bin_counts)}\n pf_peak_counts_map: {np.shape(pf_peak_counts_map)}')
                raise e
            
        return filtered_summits_analysis_df, pf_peak_counts_map

    @classmethod
    def _compute_distances_from_peaks_to_boundary(cls, active_pf_2D, filtered_flat_peaks_df, debug_print = True):
        """ Computes the distance to boundary by computing the distance to the nearest never-occupied bin
                For any given peak location, the distance to the boundary in each of the four directions can be computed.
                
            TODO: this function currently uses the binned peak positions and computes distances to the boundaries in terms of bins in each dimension. Could use a continuous position measure as well.
            

        # filtered_flat_peaks_df

        # Required Input Columns:
        # ['peak_center_binned_x', 'peak_center_binned_y']

        # Output Columns:
        # ['peak_nearest_boundary_bin_negX', 'peak_nearest_boundary_bin_posX', 'peak_nearest_boundary_bin_negY', 'peak_nearest_boundary_bin_posY'] # separate

        # ['peak_nearest_directional_boundary_bins', 'peak_nearest_directional_boundary_displacements', 'peak_nearest_directional_boundary_distances'] # combined tuple columns
        
        
        TODO: I should have just used actual continuous position values instead of counting bins :[
            
        Usage:
            peak_nearest_directional_boundary_bins, peak_nearest_directional_boundary_displacements, peak_nearest_directional_boundary_distances = _compute_distances_from_peaks_to_boundary(active_pf_2D, filtered_summits_analysis_df, debug_print=debug_print)

        """
        # Build the boundary mask from the NaN speeds, which correspond to never-occupied cells:
        # boundary_mask_indicies = ~np.isfinite(active_eloy_analysis.avg_2D_speed_per_pos)
        boundary_mask_indicies = active_pf_2D.never_visited_occupancy_mask.copy() # True if value is never-occupied, False otherwise

        ## Add a padding of size 1 of True values around the edge, ensuring a border of never-visited bins on all sides:
        # boundary_mask_indicies = np.pad(boundary_mask_indicies, 1, 'constant', constant_values=(True, True)) ## BUG: this changes the indicies and doesn't completely fix the problem

        ## Get just the True indicies. A 2-tuple of 1D np.array vectors containing the true indicies
        boundary_mask_true_indicies = np.vstack(np.where(boundary_mask_indicies)).T
        # boundary_mask_true_indicies.shape # (235, 2)
        # boundary_mask_true_indicies

        ## Compute the extrema to deal with border effects:
        # active_pf_2D.bin_info
        xbin_indicies = active_pf_2D.xbin_labels -1
        xbin_outer_extrema = (xbin_indicies[0]-1, xbin_indicies[-1]+1) # if indicies [0, 59] are valid, the outer_extrema for this axis should be (-1, 60)
        ybin_indicies = active_pf_2D.ybin_labels -1
        ybin_outer_extrema = (ybin_indicies[0]-1, ybin_indicies[-1]+1) # if indicies [0, 7] are valid, the outer_extrema for this axis should be (-1, 8)

        if debug_print:
            print(f'xbin_indicies: {xbin_indicies}\nxbin_outer_extrema: {xbin_outer_extrema}\nybin_indicies: {ybin_indicies}\nybin_outer_extrema: {ybin_outer_extrema}')
        
        peak_nearest_directional_boundary_bins, peak_nearest_directional_boundary_displacements, peak_nearest_directional_boundary_distances = list(), list(), list()

        for a_peak_row in filtered_flat_peaks_df[['peak_center_binned_x', 'peak_center_binned_y']].itertuples():
            peak_x_bin_idx, peak_y_bin_idx = (a_peak_row.peak_center_binned_x-1), (a_peak_row.peak_center_binned_y-1)
            if debug_print:
                print(f'peak_x_bin_idx: {peak_x_bin_idx}, peak_y_bin_idx: {peak_y_bin_idx}')
            # For a given (x_idx, y_idx):
            ## Perform vertical line scan (across y-values) by first getting all matching x-values:
            matching_vertical_scan_y_idxs = boundary_mask_true_indicies[(boundary_mask_true_indicies[:,0]==peak_x_bin_idx), 1] # the [*, 1] is because we only need the y-values
            # matching_vertical_scan_y_idxs # array([0, 1, 2, 6, 7], dtype=int64)
            if debug_print:
                print(f'\tmatching_vertical_scan_y_idxs: {matching_vertical_scan_y_idxs}')

            if len(matching_vertical_scan_y_idxs) == 0:
                # both min and max ends missing. Should be set to the bin just outside the minimum and maximum bin in that dimension
                warn(f'\tWARNING: len(matching_vertical_scan_y_idxs) == 0: setting matching_vertical_scan_y_idxs = {ybin_outer_extrema}')
                matching_vertical_scan_y_idxs = ybin_outer_extrema
            elif len(matching_vertical_scan_y_idxs) == 1:
                # only one end missing, need to determine which end it is and replace the missing end with the appropriate extrema
                if (matching_vertical_scan_y_idxs[0] > peak_y_bin_idx):
                    # add the lower extrema
                    warn(f'\tWARNING: len(matching_vertical_scan_y_idxs) == 1: missing lower extrema, adding ybin_outer_extrema[0] = {ybin_outer_extrema[0]} to matching_vertical_scan_y_idxs')
                    matching_vertical_scan_y_idxs = np.insert(matching_vertical_scan_y_idxs, 0, ybin_outer_extrema[0])
                    # matching_horizontal_scan_x_idxs.insert(xbin_outer_extrema[0], 0)
                elif (matching_vertical_scan_y_idxs[0] < peak_y_bin_idx):
                    # add the upper extrema
                    warn(f'\tWARNING: len(matching_vertical_scan_y_idxs) == 1: missing upper extrema, adding ybin_outer_extrema[1] = {ybin_outer_extrema[1]} to matching_vertical_scan_y_idxs')
                    matching_vertical_scan_y_idxs = np.append(matching_vertical_scan_y_idxs, [ybin_outer_extrema[1]])
                else:
                    # # EQUAL CONDITION SHOULDN'T HAPPEN!
                    # raise NotImplementedError
                    # This condition should only happen when peak_y_bin_idx is right against the boundary itself (e.g. (peak_y_bin_idx == 7) or (peak_y_bin_idx == 0)
                    if (peak_y_bin_idx == ybin_indicies[0]):
                        # matching_vertical_scan_y_idxs[0] = ybin_outer_extrema[0] ## replace the duplicated value with the lower extreme
                        warn(f'\tWARNING: peak_y_bin_idx ({peak_y_bin_idx}) == ybin_indicies[0] ({ybin_indicies[0]}): setting matching_vertical_scan_y_idxs = {ybin_outer_extrema}')
                        matching_vertical_scan_y_idxs = ybin_outer_extrema
                    elif (peak_y_bin_idx == ybin_indicies[-1]):
                        # matching_vertical_scan_y_idxs[0] = ybin_outer_extrema[1] ## replace the duplicated value with the upper extreme
                        warn(f'\tWARNING: peak_y_bin_idx ({peak_y_bin_idx}) == ybin_indicies[-1] ({ybin_indicies[-1]}): setting matching_vertical_scan_y_idxs = {ybin_outer_extrema}')
                        matching_vertical_scan_y_idxs = ybin_outer_extrema
                    else:
                        warn(f'\tWARNING: This REALLY should not happen! peak_y_bin_idx: {peak_y_bin_idx}, matching_vertical_scan_y_idxs: {matching_vertical_scan_y_idxs}!!')
                        raise NotImplementedError
                        
            ## Partition on the peak_y_bin_idx:
            found_start_indicies = np.searchsorted(matching_vertical_scan_y_idxs, peak_y_bin_idx, side='left')
            found_end_indicies = np.searchsorted(matching_vertical_scan_y_idxs, peak_y_bin_idx, side='right') # find the end of the range
            out = np.hstack((found_start_indicies, found_end_indicies))
            if debug_print:     
                print(f'\tfound_start_indicies: {found_start_indicies}, found_end_indicies: {found_end_indicies}, out: {out}')
            split_vertical_scan_y_idxs = np.array_split(matching_vertical_scan_y_idxs, [found_start_indicies]) # need to pass in found_start_indicies as a list containing the scalar value because this functionality is different than if the scalar itself is passed in.
            if debug_print:
                print(f'\tsplit_vertical_scan_y_idxs: {split_vertical_scan_y_idxs}')

            """ Encountering IndexError with split_vertical_scan_y_idxs[0][-1], says len(split_vertical_scan_y_idxs[0]) == 0
            peak_x_bin_idx: 1, peak_y_bin_idx: 0
                matching_vertical_scan_y_idxs: [6 7]
                found_start_indicies: 0, found_end_indicies: 0, out: [0 0]
                split_vertical_scan_y_idxs: [array([], dtype=int64), array([6, 7], dtype=int64)]

            """
            lower_list, upper_list = split_vertical_scan_y_idxs[0], split_vertical_scan_y_idxs[1]
            if len(lower_list)==0:
                # if the lower list is empty get the ybin_outer_extrema[0]
                below_bound = ybin_outer_extrema[0]
            else:
                below_bound = lower_list[-1] # get the last (maximum) of the lower list

            if len(upper_list)==0:
                # if the upper list is empty get the ybin_outer_extrema[1]
                above_bound = ybin_outer_extrema[1]
            else:
                above_bound = upper_list[0] # get the first (minimum) of the upper list
            vertical_scan_result = (below_bound, above_bound) # get the last (maximum) of the lower list, and the first (minimum) of the upper list.
            if debug_print:
                print(f'\tvertical_scan_result: {vertical_scan_result}') # vertical_scan_result: (2, 6)


            ## Perform horizontal line scan (across x-values):
            matching_horizontal_scan_x_idxs = boundary_mask_true_indicies[(boundary_mask_true_indicies[:,1]==peak_y_bin_idx), 0] # the [*, 0] is because we only need the x-values
            # matching_horizontal_scan_x_idxs # array([0, 1, 2, 6, 7], dtype=int64)
            if debug_print:
                print(f'\tmatching_horizontal_scan_x_idxs: {matching_horizontal_scan_x_idxs}')

            if len(matching_horizontal_scan_x_idxs) == 0:
                # both min and max ends missing. Should be set to the bin just outside the minimum and maximum bin in that dimension
                warn(f'\tWARNING: len(matching_horizontal_scan_x_idxs) == 0: setting matching_horizontal_scan_x_idxs = {xbin_outer_extrema}')
                matching_horizontal_scan_x_idxs = xbin_outer_extrema
            elif len(matching_horizontal_scan_x_idxs) == 1:
                # only one end missing, need to determine which end it is and replace the missing end with the appropriate extrema
                if (matching_horizontal_scan_x_idxs[0] > peak_x_bin_idx):
                    # add the lower extrema
                    warn(f'\tWARNING: len(matching_horizontal_scan_x_idxs) == 1: missing lower extrema, adding xbin_outer_extrema[0] = {xbin_outer_extrema[0]} to matching_horizontal_scan_x_idxs')
                    matching_horizontal_scan_x_idxs = np.insert(matching_horizontal_scan_x_idxs, 0, xbin_outer_extrema[0])
                    # matching_horizontal_scan_x_idxs.insert(xbin_outer_extrema[0], 0)
                elif (matching_horizontal_scan_x_idxs[0] < peak_x_bin_idx):
                    # add the upper extrema
                    warn(f'\tWARNING: len(matching_horizontal_scan_x_idxs) == 1: missing upper extrema, adding xbin_outer_extrema[1] = {xbin_outer_extrema[1]} to matching_horizontal_scan_x_idxs')
                    matching_horizontal_scan_x_idxs = np.append(matching_horizontal_scan_x_idxs, [xbin_outer_extrema[1]])
                else:
                    # # EQUAL CONDITION SHOULDN'T HAPPEN!
                    # raise NotImplementedError
                    # This condition should only happen when peak_x_bin_idx is right against the boundary itself (e.g. (peak_x_bin_idx == 7) or (peak_x_bin_idx == 0)
                    if (peak_x_bin_idx == xbin_indicies[0]):
                        # matching_horizontal_scan_x_idxs[0] = xbin_outer_extrema[0] ## replace the duplicated value with the lower extreme
                        warn(f'\tWARNING: peak_x_bin_idx ({peak_x_bin_idx}) == xbin_indicies[0] ({xbin_indicies[0]}): setting matching_horizontal_scan_x_idxs = {xbin_outer_extrema}')
                        matching_horizontal_scan_x_idxs = xbin_outer_extrema
                    elif (peak_x_bin_idx == xbin_indicies[-1]):
                        # matching_horizontal_scan_x_idxs[0] = xbin_outer_extrema[1] ## replace the duplicated value with the upper extreme
                        warn(f'\tWARNING: peak_x_bin_idx ({peak_x_bin_idx}) == xbin_indicies[-1] ({xbin_indicies[-1]}): setting matching_horizontal_scan_x_idxs = {xbin_outer_extrema}')
                        matching_horizontal_scan_x_idxs = xbin_outer_extrema
                    else:
                        warn(f'\tWARNING: This REALLY should not happen! peak_x_bin_idx: {peak_x_bin_idx}, matching_horizontal_scan_x_idxs: {matching_horizontal_scan_x_idxs}!!')
                        raise NotImplementedError
                        
            # Otherwise we're good

            ### Partition on the peak_x_bin_idx
            found_start_indicies = np.searchsorted(matching_horizontal_scan_x_idxs, peak_x_bin_idx, side='left')
            found_end_indicies = np.searchsorted(matching_horizontal_scan_x_idxs, peak_x_bin_idx, side='right') # find the end of the range
            out = np.hstack((found_start_indicies, found_end_indicies))
            if debug_print:
                print(f'\tfound_start_indicies: {found_start_indicies}, found_end_indicies: {found_end_indicies}, out: {out}')
            split_horizontal_scan_x_idxs = np.array_split(matching_horizontal_scan_x_idxs, [found_start_indicies]) # need to pass in found_start_indicies as a list containing the scalar value because this functionality is different than if the scalar itself is passed in.
            if debug_print:
                print(f'\tsplit_horizontal_scan_x_idxs: {split_horizontal_scan_x_idxs}')

            lower_list, upper_list = split_horizontal_scan_x_idxs[0], split_horizontal_scan_x_idxs[1]
            if len(lower_list)==0:
                # if the lower list is empty get the xbin_outer_extrema[0]
                below_bound = xbin_outer_extrema[0]
            else:
                below_bound = lower_list[-1] # get the last (maximum) of the lower list
            if len(upper_list)==0:
                # if the upper list is empty get the xbin_outer_extrema[1]
                above_bound = xbin_outer_extrema[1]
            else:
                above_bound = upper_list[0] # get the first (minimum) of the upper list
            horizontal_scan_result = (below_bound, above_bound) # get the last (maximum) of the lower list, and the first (minimum) of the upper list.
            if debug_print:
                print(f'\thorizontal_scan_result: {horizontal_scan_result}') # horizontal_scan_result: (0, 60)
            
            ## Build final four directional boundary bins:
            final_four_boundary_bin_tuples = [(peak_x_bin_idx, boundary_y) for boundary_y in vertical_scan_result] # [(46, 2), (46, 7)]
            final_four_boundary_bin_tuples += [(boundary_x, peak_y_bin_idx) for boundary_x in horizontal_scan_result] # [(0, 4), (60, 4)]
            # final_four_boundary_bin_tuples # [(46, 2), (46, 7), (0, 4), (60, 4)]
            # Add to outputs:
            peak_nearest_directional_boundary_bins.append(final_four_boundary_bin_tuples)
            final_four_boundary_bins = np.array(final_four_boundary_bin_tuples) # convert to a (4, 2) np.array
            if debug_print:
                print(f'\tfinal_four_boundary_bins: {final_four_boundary_bins}')
            ## Compute displacements from current point to each boundary:
            final_four_boundary_displacements = final_four_boundary_bins - [peak_x_bin_idx, peak_y_bin_idx]
            if debug_print:
                print(f'\tfinal_four_boundary_displacements: {final_four_boundary_displacements}')

            # Add to outputs:
            peak_nearest_directional_boundary_displacements.append([(final_four_boundary_displacements[row_idx,0], final_four_boundary_displacements[row_idx,1]) for row_idx in np.arange(final_four_boundary_displacements.shape[0])])

            # Compute distances from current point to each boundary:
            # Flatten down to the pure distances in each component axis, form is (down, up, left, right)
            final_four_boundary_distances = np.max(np.abs(final_four_boundary_displacements), axis=1) # array([ 2,  2, 47, 14], dtype=int64)
            # final_four_boundary_distances # again a (4, 2) np.array
            if debug_print:
                print(f'\tfinal_four_boundary_distances: {final_four_boundary_distances}')
            peak_nearest_directional_boundary_distances.append(final_four_boundary_distances)
        
        return peak_nearest_directional_boundary_bins, peak_nearest_directional_boundary_displacements, peak_nearest_directional_boundary_distances



    @classmethod
    def _perform_find_posterior_peaks_peak_prominence2d_computation(cls, p_x_given_n_list, xbin_centers, ybin_centers, step=0.01, peak_height_multiplier_probe_levels=(0.5, 0.9), minimum_included_peak_height = 0.2, uniform_blur_size = 3, gaussian_blur_sigma = 3, debug_print=False):
            """ Uses the peak_prominence2d package to find the peaks and promenences of 2D placefields
            
            Independent of the other peak-computing computation functions above
            
            Inputs:
                peak_height_multiplier_probe_levels = (0.5, 0.9) # 50% and 90% of the peak height
                gaussian_blur_sigma: input to gaussian_filter for blur of peak counts
                uniform_blur_size: inputt to uniform_filter for blur of peak counts
                
                
            Requires:
                computed_data['pf2D']
                
            Provides:
                computed_data['RatemapPeaksAnalysis']['PeakProminence2D']:
                    computed_data['RatemapPeaksAnalysis']['PeakProminence2D']['xx']:
                    computed_data['RatemapPeaksAnalysis']['PeakProminence2D']['yy']:
                    computed_data['RatemapPeaksAnalysis']['PeakProminence2D']['neuron_extended_ids']
                    
                    
                    computed_data['RatemapPeaksAnalysis']['PeakProminence2D']['result_tuples']: (slab, peaks, idmap, promap, parentmap)
                    
                    flat_peaks_df
                    filtered_flat_peaks_df
                    
                    peak_counts
                        raw
                        uniform_blurred
                        gaussian_blurred
                    
            """
            n_epochs = len(p_x_given_n_list)
            
            ## TODO: change to amap with keys of active_pf_2D.neuron_ids
            
            # active_tuning_curves = active_pf_2D.ratemap.tuning_curves # Raw Tuning Curves
            active_tuning_curves = active_pf_2D.ratemap.unit_max_tuning_curves # Unit-max scaled tuning curves
            # tuning_curve_peak_firing_rates = active_pf_2D.ratemap.tuning_curve_peak_firing_rates # the peak firing rates of each tuning curve
            tuning_curve_peak_firing_rates = active_pf_2D.ratemap.tuning_curve_unsmoothed_peak_firing_rates
             
            #  Build the results:
            out_results = {}
            out_cell_peak_dfs_list = []
            n_slices = len(peak_height_multiplier_probe_levels)

            for epoch_idx in np.arange(n_epochs):
                p_x_given_n = p_x_given_n_list[epoch_idx]

                a_p_x_given_n = np.squeeze(p_x_given_n[:, :, a_t_bin_idx])

                slab = a_p_x_given_n.T
                _, _, slab, a_t_bin_peaks_dict, id_map, prominence_map, parent_map = compute_prominence_contours(xbin_centers=xbin_centers, ybin_centers=ybin_centers, slab=slab, step=step, min_area=None, min_depth=0.2, include_edge=True, verbose=False)
                #""" Analyze all peaks of a given cell/ratemap """
                n_peaks = len(a_t_bin_peaks_dict)
                
                ## Neuron:
                n_total_cell_slice_results = n_slices * n_peaks                
                neuron_id_arr = np.full((n_total_cell_slice_results,), neuron_id) # repeat the neuron_id many times for the datatable
                neuron_peak_curve_rate_arr = np.full((n_total_cell_slice_results,), neuron_tuning_curve_peak_firing_rate) # repeat the neuron_id many times for the datatable
                
                ## Peak
                summit_slice_peak_id_arr = np.zeros((n_peaks, n_slices), dtype=np.int16) # same summit/peak id for all in the slice
                summit_slice_peak_level_multiplier_arr = np.zeros((n_peaks, n_slices), dtype=float) # same summit/peak id for all in the slice
                summit_slice_peak_level_arr = np.zeros((n_peaks, n_slices), dtype=float) # same summit/peak id for all in the slice
                summit_slice_peak_height_arr = np.zeros((n_peaks, n_slices), dtype=float) # same summit/peak id for all in the slice
                summit_slice_peak_prominence_arr = np.zeros((n_peaks, n_slices), dtype=float) # same summit/peak id for all in the slice
                summit_peak_center_x_arr = np.zeros((n_peaks, n_slices), dtype=float)
                summit_peak_center_y_arr = np.zeros((n_peaks, n_slices), dtype=float)
                
                ## Slice
                summit_slice_idx_arr = np.tile(np.arange(n_slices), n_peaks).astype('int') # array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
                summit_slice_x_side_length_arr = np.zeros((n_peaks, n_slices), dtype=float)
                summit_slice_y_side_length_arr = np.zeros((n_peaks, n_slices), dtype=float)
                summit_slice_center_x_arr = np.zeros((n_peaks, n_slices), dtype=float)
                summit_slice_center_y_arr = np.zeros((n_peaks, n_slices), dtype=float)

                for peak_idx, (peak_id, a_peak_dict) in enumerate(a_t_bin_peaks_dict.items()):
                    if debug_print:
                        print(f'computing contours for peak_id: {peak_id}...')                    
                    summit_slice_peak_height_arr[peak_idx, :] = a_peak_dict['height']
                    summit_slice_peak_prominence_arr[peak_idx, :] = a_peak_dict['prominence']
                    summit_peak_center_x_arr[peak_idx, :] = a_peak_dict['center'][0]
                    summit_peak_center_y_arr[peak_idx, :] = a_peak_dict['center'][1]
                    
                    ## This is where we would loop through each desired slice/probe levels:
                    a_peak_dict['probe_levels'] = np.array([a_peak_dict['height']*multiplier for multiplier in peak_height_multiplier_probe_levels]).astype('float') # specific probe levels
                    summit_slice_peak_level_multiplier_arr[peak_idx, :] = np.array(peak_height_multiplier_probe_levels).astype('float')
                    summit_slice_peak_level_arr[peak_idx, :] = a_peak_dict['probe_levels']
                    included_computed_contours = PeakPromenence._find_contours_at_levels(xbin_centers, ybin_centers, slab, a_peak_dict['center'], a_peak_dict['probe_levels']) # DONE: efficiency: This would be more efficient to do for all peaks at once I believe. CONCLUSION: No, this needs to be done separately for each peak as they each have separate prominences which determine the levels they should be sliced at.
                    ## Build the dict that contains the output level slices
                    a_peak_dict['level_slices'] = {probe_lvl:{'contour':contour, 'bbox':contour.get_extents(), 'size':contour.get_extents().size} for probe_lvl, contour in included_computed_contours.items() if (contour is not None)} # if contour is None, it looks like the 'build flat output' step fails below

                    if debug_print:
                        print(f"probe_levels: {a_peak_dict['probe_levels']}")

                    ## Build flat output:
                    for lvl_idx, probe_lvl in enumerate(a_peak_dict['probe_levels']):
                        # a_slice = a_peak_dict['level_slices'][probe_lvl]
                        a_slice = a_peak_dict['level_slices'].get(probe_lvl, None) # allow missing entries. This will occur when (contour is None) above.
                        #TODO 2023-09-25 23:59: - [ ] Do I need to do anything else to handle this case, like remove the invalid curve from 
                        if a_slice is None:
                            print(f'WARNING: a_slice is None. 2023-09-25 - Unsure if this is okay, used to be a fatal error.') # a_peak_dict: {a_peak_dict}
                        else:
                            slice_bbox = a_slice['bbox']
                            (x0, y0, width, height) = slice_bbox.bounds        
                            summit_slice_peak_id_arr[peak_idx, lvl_idx] = peak_id
                            summit_slice_x_side_length_arr[peak_idx, lvl_idx] = width
                            summit_slice_y_side_length_arr[peak_idx, lvl_idx] = height
                            # summit_slice_center_x_arr[peak_idx, lvl_idx] = slice_bbox.center[0]
                            # summit_slice_center_y_arr[peak_idx, lvl_idx] = slice_bbox.center[1]
                            summit_slice_center_x_arr[peak_idx, lvl_idx] = float(x0) + (0.5 * float(width))
                            summit_slice_center_y_arr[peak_idx, lvl_idx] = float(y0) + (0.5 * float(height))
                    
                if debug_print:
                    print(f'building peak_df for neuron[{epoch_idx}] with {n_peaks}...')
                cell_peaks_df = pd.DataFrame({'neuron_id': neuron_id_arr, 'neuron_peak_firing_rate': neuron_peak_curve_rate_arr, 'summit_idx': summit_slice_peak_id_arr.flatten(), 'summit_slice_idx': summit_slice_idx_arr.flatten(),
                                             'slice_level_multiplier': summit_slice_peak_level_multiplier_arr.flatten(), 'summit_slice_level': summit_slice_peak_level_arr.flatten(),
                                             'peak_relative_height': summit_slice_peak_height_arr.flatten(), 'peak_prominence': summit_slice_peak_prominence_arr.flatten(),
                                             'peak_center_x': summit_peak_center_x_arr.flatten(), 'peak_center_y': summit_peak_center_y_arr.flatten(),
                                             'summit_slice_x_width': summit_slice_x_side_length_arr.flatten(), 'summit_slice_y_width': summit_slice_y_side_length_arr.flatten(),
                                             'summit_slice_center_x': summit_slice_center_x_arr.flatten(), 'summit_slice_center_y': summit_slice_center_y_arr.flatten()
                                             })
                cell_peaks_df['peak_height'] = cell_peaks_df['peak_relative_height'] * neuron_peak_curve_rate_arr
                
                out_cell_peak_dfs_list.append(cell_peaks_df)
                
                                           
                if debug_print:
                    print(f'done.') # END Analyze peaks
                    
                out_results[neuron_id] = {'peaks': a_t_bin_peaks_dict, 'slab': slab, 'id_map':id_map, 'prominence_map':prominence_map, 'parent_map':parent_map} 
    
            # Build final concatenated dataframe:
            if debug_print:
                print(f'building final concatenated cell_peaks_df for {n_epochs} total neurons...')
            cell_peaks_df = pd.concat(out_cell_peak_dfs_list)
            ## Find which position bin each peak falls in and add it to the flat_peaks_df:
            cell_peaks_df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(cell_peaks_df, bin_values=(xbin, ybin), position_column_names=('peak_center_x', 'peak_center_y'), binned_column_names=('peak_center_binned_x', 'peak_center_binned_y'), active_computation_config=None, force_recompute=False, debug_print=debug_print)

            ## Find the avg velocity corresponding to each position bin containing a peak value:
            active_eloy_analysis = computation_result.computed_data.get('EloyAnalysis', None)            
            if active_eloy_analysis is not None:
                ## Extract the previously computed results:
                avg_2D_speed_per_pos = active_eloy_analysis.avg_2D_speed_per_pos # (60, 8)
                # avg_2D_speed_sort_idxs = active_eloy_analysis.avg_2D_speed_sort_idxs
                # assert np.shape(avg_2D_speed_per_pos) == np.shape(pf_peak_counts_map), f"the shape of the active_eloy_analysis.avg_2D_speed_per_pos ({np.shape(avg_2D_speed_per_pos)}) and the shape of the newly built pf_peak_counts_map ({np.shape(pf_peak_counts_map)}) must match!"
                _temp_peak_center_bin_label_xy = cell_peaks_df[['peak_center_binned_x', 'peak_center_binned_y']].to_numpy()-1 # (26, 2)
                ## Add the column to matrix:
                cell_peaks_df['peak_center_avg_speed'] = avg_2D_speed_per_pos[_temp_peak_center_bin_label_xy[:,0], _temp_peak_center_bin_label_xy[:,1]] # array([33.2289, 33.7748, 35.9964, 0, 59.3887, 37.354, 16.1506, 0, 49.8418, 33.2289, 0, 14.9843, 33.8302, 29.8891, 37.354, 10.6905, 0, 33.7748, nan, 53.9505, 34.003, 33.7748, 22.7252, nan, 11.8898, 58.9018])                
            

            ## Filter the summits, compute velocities, etc:            
            filtered_summits_analysis_df, pf_peak_counts_map = PeakPromenence._build_filtered_summits_analysis_results(xbin, ybin, xbin_labels, ybin_labels, cell_peaks_df, active_eloy_analysis, slice_level_multiplier=0.5, minimum_included_peak_height=minimum_included_peak_height, debug_print = debug_print)
            
            pf_peak_counts_map_blurred = uniform_filter(pf_peak_counts_map.astype('float'), size=uniform_blur_size, mode='constant')
            pf_peak_counts_map_blurred_gaussian = gaussian_filter(pf_peak_counts_map.astype('float'), sigma=gaussian_blur_sigma)
            pf_peak_counts_results = DynamicParameters(raw=pf_peak_counts_map, uniform_blurred=pf_peak_counts_map_blurred, gaussian_blurred=pf_peak_counts_map_blurred_gaussian)

            try:
                ## Add distance to boundary by computing the distance to the nearest never-occupied bin
                peak_nearest_directional_boundary_bins, peak_nearest_directional_boundary_displacements, peak_nearest_directional_boundary_distances = PeakPromenence._compute_distances_from_peaks_to_boundary(active_pf_2D, filtered_summits_analysis_df, debug_print=debug_print)

                ## Add the output columns to the peaks dataframe:
                # Output Columns:
                # ['peak_nearest_boundary_bin_negX', 'peak_nearest_boundary_bin_posX', 'peak_nearest_boundary_bin_negY', 'peak_nearest_boundary_bin_posY'] # separate
                # ['peak_nearest_directional_boundary_bins', 'peak_nearest_directional_boundary_displacements', 'peak_nearest_directional_boundary_distances'] # combined tuple columns
                filtered_summits_analysis_df['peak_nearest_directional_boundary_bins'] = peak_nearest_directional_boundary_bins
                filtered_summits_analysis_df['peak_nearest_directional_boundary_displacements'] = peak_nearest_directional_boundary_displacements
                filtered_summits_analysis_df['peak_nearest_directional_boundary_distances'] = peak_nearest_directional_boundary_distances
                filtered_summits_analysis_df['nearest_directional_boundary_direction_idx'] = np.argmin(peak_nearest_directional_boundary_distances, axis=1) # an index [0,1,2,3] corresponding to the direction of travel to the nearest index. Corresponds to (down, up, left, right)
                filtered_summits_analysis_df['nearest_directional_boundary_direction_distance'] = np.min(peak_nearest_directional_boundary_distances, axis=1) # the distance in the minimal dimension towards the nearest boundary

                # ['peak_nearest_boundary_bin_negX', 'peak_nearest_boundary_bin_posX', 'peak_nearest_boundary_bin_negY', 'peak_nearest_boundary_bin_posY'] # separate
                distances = np.vstack([np.asarray(a_tuple) for a_tuple in peak_nearest_directional_boundary_distances])
                x_distances = np.min(distances[:,3:], axis=1) # find the distance to nearest wall vertically
                y_distances = np.min(distances[:,:2], axis=1) # find the distance to nearest wall horizontally

                filtered_summits_analysis_df['nearest_x_boundary_distance'] = x_distances # the distance in the minimal dimension towards the nearest x boundary
                filtered_summits_analysis_df['nearest_y_boundary_distance'] = y_distances # the distance in the minimal dimension towards the nearest y boundary
            except (BaseException, NotImplementedError) as err:
                print(f'could not find distances to nearest boundary in `ratemap_peaks_prominence2d`. Some columns will be missing from the output dataframe. Error: {err}')

                            
            return DynamicParameters(xx=xbin_centers, yy=ybin_centers, results=out_results, flat_peaks_df=cell_peaks_df, filtered_flat_peaks_df=filtered_summits_analysis_df, peak_counts=pf_peak_counts_results)



















#-------------------Plot------------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_Prominence(xx, yy, slab, peaks, idmap, promap, parentmap, n_contour_levels=None, debug_print=False):
    """ simple test plot of the results calculated from getProminence.
    
    Inputs:
        n_contour_levels: should be an integer indicating the number of levels to display in the contour plot
        
    Usage:
    
        from pyphoplacecellanalysis.External.peak_prominence2d import getProminence, plot_Prominence
        
        step = 0.2
        xx = active_pf_2D_dt.xbin_labels
        yy = active_pf_2D_dt.ybin_labels
        slab = active_pf_2D.ratemap.tuning_curves[3].T
        zmax = slab.max()
        peaks, idmap, promap, parentmap = getProminence(slab, step, ybin_centers=yy, xbin_centers=xx, min_area=None, include_edge=True, verbose=False)
        figure, (ax1, ax2, ax3, ax4) = plot_Prominence(xx, yy, slab, peaks, idmap, promap, parentmap, debug_print=False)
    
    """
    figure = plt.figure(figsize=(12,10),dpi=100)
    zmax = slab.max()
    XX, YY = np.meshgrid(xx, yy)

    # ==================================================================================================================== #
    ## Subplot 1: Top-Left - Contour Plot
    ax1=figure.add_subplot(2,2,1)
    
    if n_contour_levels is not None:
        levels = np.linspace(0.0, zmax, n_contour_levels)
    else:
        levels = np.arange(0, zmax, 1) # old way
    ax1.contourf(XX, YY, slab, levels=levels)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Top view, col contours as dashed lines')

    # This plots the dashed lines on top of the contour plot, but idk what the dashed lines even are. They're often out in space irrelevant to the main peaks.
    # The dotted black lines refer to the "col"s (see definition of col in header) of the peaks. 
    for key, value in peaks.items():
        if debug_print:
            print (key)
        cols=value['contour']
        ax1.plot(cols.vertices[:,0], cols.vertices[:,1],'k:')

    # ==================================================================================================================== #
    ## Subplot 2: Top-Right - Cross-section
    line=slab[slab.shape[0]//2]
    ax2=figure.add_subplot(2,2,2)
    ax2.plot(xx,line,'b-')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('Cross section through y=0')

    # This adds the vertical black dotted lines to the cross-section through each peak and the text with the peak label/parent
    for key, value in peaks.items():
        xii, yii = value['center']
        z2ii = value['height']
        pro = value['prominence']
        z1ii = z2ii-pro
        ax2.plot([xii, xii], [z1ii, z2ii],'k:')
        ax2.text(xii, z2ii,'p%d, parent = %d' %(key, value['parent']),
                horizontalalignment='center',
                verticalalignment='bottom')

    # ==================================================================================================================== #
    ## Subplot 3: Bottom-Left - 3D Grid
    ax3=figure.add_subplot(2,2,3,projection='3d')
        
    # this actually plots the 3D surface:
    ax3.plot_surface(XX, YY, slab, rstride=4, cstride=4, cmap='viridis', alpha=0.8) 
    # rstride, cstride: Downsampling stride in each direction. These arguments are mutually exclusive with rcount and ccount.
    
    
    ## This part looks like it just plots some ascending vertical lines through the peaks of the 3D plot, but you can't really see them. They look like they go through the center of the peak.
    for key, value in peaks.items():
        xii,yii=value['center']
        z2ii=value['height']
        pro=value['prominence']
        z1ii=z2ii-pro
        ax3.plot([xii,xii],[yii,yii],[z1ii,z2ii], color='r', linewidth=2)
        
        

    # ==================================================================================================================== #
    ## Subplot 4: Bottom-Right - Matrix of Promeneces
    ax4=figure.add_subplot(2,2,4)
    cs=ax4.imshow(promap,origin='lower',interpolation='nearest', extent=[-10,10,-10,10])
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('Top view, prominences at peaks')
    plt.colorbar(cs,ax=ax4)

    plt.show(block=False)

    if debug_print:
        from pprint import pprint
        pprint(peaks)
    
    return figure, (ax1, ax2, ax3, ax4)







#-------------Main---------------------------------
if __name__=='__main__':

    #------------------A toy example------------------
    xx=np.linspace(-10,10,100)
    yy=np.linspace(-10,10,100)

    XX,YY=np.meshgrid(xx,yy)
    slab=np.zeros(XX.shape)

    # add 3 peaks
    slab+=5*np.exp(-XX**2/1**2 - YY**2/1**2)
    slab+=8*np.exp(-(XX-3)**2/2**2 - YY**2/2**2)
    slab+=10*np.exp(-(XX+4)**2/2**2 - YY**2/2**2)

    step=0.2
    peaks, idmap, promap, parentmap = getProminence(slab, step, ybin_centers=yy, xbin_centers=xx, min_area=None, include_edge=True, verbose=False)

    #-------------------Plot------------------------
    figure, (ax1, ax2, ax3, ax4) = plot_Prominence(xx, yy, slab, peaks, idmap, promap, parentmap, debug_print=False)
    
    from pprint import pprint
    pprint(peaks)

    figure.show()
    plt.show()
