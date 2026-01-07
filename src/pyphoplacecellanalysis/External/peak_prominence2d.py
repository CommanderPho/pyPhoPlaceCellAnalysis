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
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types
import nptyping as ND
from nptyping import NDArray
import numpy as np

from matplotlib.transforms import Bbox
from matplotlib.path import Path
import matplotlib.pyplot as plt
# At module level or start of functions
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
plt.ioff()  # Disable interactive mode

from scipy.interpolate import interp1d
from warnings import warn
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from attrs import define, field
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputedResult
from neuropy.utils.mixins.AttrsClassHelpers import serialized_field, non_serialized_field
from neuropy.utils.mixins.indexing_helpers import get_dict_subset

DecodedEpochIndex: TypeAlias = int # an integer index that is an aclu
DecodedEpochTimeBinIndex: TypeAlias = int # an integer index that is an aclu

# Define a new type as a tuple of the two above custom types
DecodedEpochTimeBinIndexTuple: TypeAlias = Tuple[DecodedEpochIndex, DecodedEpochTimeBinIndex]


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


def getProminence(var, step, ybin_centers=None, xbin_centers=None, min_considered_promenence=None, include_edge=True, min_area=None, max_area=None, area_func=contourArea, centroid_num_to_center=5, allow_hole=True, max_hole_area=None, verbose=False):
    '''Find 2d prominences of peaks.

    <var>: 2D ndarray, data to find local maxima. Missings (nans) are masked.
    <step>: float, contour interval. Finer (smaller) interval gives better accuarcy.
    <ybin_centers>, <xbin_centers>: 1d array, y and x coordinates of <var>. If not given,
                    use int indices.
    <min_considered_promenence>: float, filter out peaks with prominence smaller than this.
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
            'col_level' : height level (altitude) at col (saddle),
            'prominence': prominence of peak,
            'area'      : float, area of col contour. If latitude and 
                          longitude axes available, geographical area in
                          km^2. Otherwise, area in unit^2, unit is the same
                          as x, y axes,
            'contours'  : list, contours of peak from peak summit down to col (saddle),
                          each being a matplotlib Path obj
            'parent'    : int, id of a peak's parent. Heightest peak as a
                          parent id of 0.

    Author: guangzhi XU (xugzhi1987@gmail.com; guangzhi.xu@outlook.com)
    Update time: 2018-11-11 18:42:04.
    '''
    # At module level or start of functions
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    plt.ioff()  # Disable interactive mode

    # Create figure once, reuse it
    fig, ax = plt.subplots()
    ax.set_axis_off()  # Disable axis rendering to reduce overhead

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
            # Remove ax.cla() - it's expensive and not needed
            # ax.cla()
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
                    if min_considered_promenence is not None and len(match_list)>1:
                        match_list2=[]
                        for mm in match_list:
                            if prominence[mm]<min_considered_promenence:
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

    ## END for ii,levii in enumerate(levels[::-1])...

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
        nc: int = min(centroid_num_to_center,len(vv))
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
    ## END for ii in range(len(peaks))
    
    plt.close(fig)

    return result, id_map, result_map, parent_map ## result -> peaks_dict, result_map -> prominence_map


def compute_prominence_contours(xbin_centers: NDArray, ybin_centers: NDArray, slab: NDArray, step: float=0.1, min_area: Optional[float]=None, min_considered_promenence: float=0.2, include_edge: bool=True, verbose: bool=False, **kwargs):
    """ Simple wrapper around the getProminence function by Pho Hale
    xbin_centers and ybin_centers should be like *bin_labels not *bin
    slab should usually be transposed: tuning_curves[i].T
    
    Usage:        
        step = 0.2
        i = 0
        xx, yy, slab, peaks, idmap, promap, parentmap = perform_compute_prominence_contours(active_pf_2D_dt.xbin_labels, active_pf_2D_dt.ybin_labels, active_pf_2D.ratemap.tuning_curves[i].T, step=step)
        
        # Test plot the promenence result
        figure, (ax1, ax2, ax3, ax4) = PeakPromenenceDisplay.plot_Prominence(xx, yy, slab, peaks, idmap, promap, parentmap, debug_print=False)

    """
    peaks_dict, id_map, prominence_map, parent_map = getProminence(slab, step, ybin_centers=ybin_centers, xbin_centers=xbin_centers, min_area=min_area, min_considered_promenence=min_considered_promenence, include_edge=include_edge, verbose=verbose, **kwargs)
    return xbin_centers, ybin_centers, slab, peaks_dict, id_map, prominence_map, parent_map


from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from scipy import ndimage # used for `PeakPromenence.compute_2d_peak_prominence`
from skimage.morphology import reconstruction # used for `PeakPromenence.compute_2d_peak_prominence`

@define(slots=False, repr=False, eq=False)
class SlabResult(ComputedResult):
    """Simple attrs class to hold slab result information.

    from pyphoplacecellanalysis.External.peak_prominence2d import SlabResult

    a_slab_result_dict = {
        'peaks': peaks_dict,
        'slab': slab,
        'id_map': id_map,
        'prominence_map': prominence_map,
        'parent_map': parent_map
    }

    slab_result: SlabResult = SlabResult(**a_slab_result_dict)


    slab_result: SlabResult = SlabResult(
        peaks=peaks_dict,
        slab=slab,
        id_map=id_map,
        prominence_map=prominence_map,
        parent_map=parent_map
    )
    
    peaks:

        peakii={
            'id'         : kk, # int - 1
            'height'  : np.max(lev_range), # float - 0.11800000000000001
            'col_level'  : np.min(lev_range), # float - 0.0
            'prominence'  : prokk, # float - 0.11800000000000001
            'area'       : vv[-1].area, # float - 37146.31290118136
            'contours'   : vv, # contours: List[matplotlib.path.Path]
            'contour'    : vv[-1], # contour: matplotlib.path.Path
            'center'     : centerkk, # NDArray - [90.8969 38.232] - (2,)
            'parent'     : parents[kk] # int - 0
            }
            

    """
    peaks: Dict[int, Dict] = serialized_field()
    slab: NDArray[ND.Shape["N_X_BINS, N_Y_BINS"], np.floating] = serialized_field()
    id_map: NDArray[ND.Shape["N_X_BINS, N_Y_BINS"], np.floating] = serialized_field()
    prominence_map: NDArray[ND.Shape["N_X_BINS, N_Y_BINS"], np.floating] = serialized_field()
    parent_map: NDArray[ND.Shape["N_X_BINS, N_Y_BINS"], np.floating] = serialized_field()

    def __repr__(self):
        from pyphocorehelpers.print_helpers import strip_type_str_to_classname
        attr_reprs = []
        for a in self.__attrs_attrs__:
            attr_type = strip_type_str_to_classname(type(getattr(self, a.name)))
            value = getattr(self, a.name)
            if hasattr(value, 'shape'):
                attr_reprs.append(f"{a.name}: {attr_type} | shape {value.shape}")
            else:
                attr_reprs.append(f"{a.name}: {attr_type}")
        content = ",\n\t".join(attr_reprs)
        return f"{type(self).__name__}({content}\n)"




@define(slots=False, repr=False, eq=False)
class PeakCounts(ComputedResult):
    """Nested class containing raw and blurred peak count maps."""
    _VersionedResultMixin_version: str = "2026.01.05_0"

    raw: NDArray = serialized_field()
    uniform_blurred: NDArray = serialized_field()
    gaussian_blurred: NDArray = serialized_field()

    # For serialization/pickling: ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        # # Remove the unpicklable entries.
        # _non_pickled_fields = ['curr_active_pipeline', 'track_templates']
        # for a_non_pickleable_field in _non_pickled_fields:
        #     del state[a_non_pickleable_field]
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        # For `VersionedResultMixin`
        self._VersionedResultMixin__setstate__(state)

        # _non_pickled_field_restore_defaults = dict(zip(['curr_active_pipeline', 'track_templates'], [None, None]))
        # for a_field_name, a_default_restore_value in _non_pickled_field_restore_defaults.items():
        #     if a_field_name not in state:
        #         state[a_field_name] = a_default_restore_value

        self.__dict__.update(state)
        # # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        # super(PeakCounts, self).__init__() # from

    def __repr__(self):
        """ 2024-01-11 - Renders only the fields and their sizes
        """
        from pyphocorehelpers.print_helpers import strip_type_str_to_classname
        attr_reprs = []
        for a in self.__attrs_attrs__:
            attr_type = strip_type_str_to_classname(type(getattr(self, a.name)))
            if 'shape' in a.metadata:
                shape = ', '.join(a.metadata['shape'])  # this joins tuple elements with a comma, creating a string without quotes
                attr_reprs.append(f"{a.name}: {attr_type} | shape ({shape})")  # enclose the shape string with parentheses
            else:
                attr_reprs.append(f"{a.name}: {attr_type}")
        content = ",\n\t".join(attr_reprs)
        return f"{type(self).__name__}({content}\n)"

    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, debug_print=False, enable_hdf_testing_mode:bool=False, OVERRIDE_ALLOW_GLOBAL_NESTED_EXPANSION:bool=None, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, debug_print=debug_print, enable_hdf_testing_mode=enable_hdf_testing_mode, OVERRIDE_ALLOW_GLOBAL_NESTED_EXPANSION=OVERRIDE_ALLOW_GLOBAL_NESTED_EXPANSION, **kwargs)


    @classmethod
    def _reload_class(cls, an_instance):
        """ specifically updates the instance after its class definition has been updated.
        """
        non_init_subset=['_VersionedResultMixin_version']

        _full_state = an_instance.__getstate__()
        _init_state = get_dict_subset(_full_state, subset_excludelist=non_init_subset)
        _post_init_state = get_dict_subset(_full_state, subset_includelist=non_init_subset)
        _obj = cls(**_init_state)
        _obj.__dict__.update(**_post_init_state) ## perform literal update
        return _obj


@define(slots=False, repr=False, eq=False)
class PosteriorPeaksPeakProminence2dResult(ComputedResult):
    """Result class for posterior peaks peak prominence 2D computation."""
    _VersionedResultMixin_version: str = "2026.01.05_0"

    xx: NDArray = serialized_field()
    yy: NDArray = serialized_field()
    results: Dict[DecodedEpochTimeBinIndexTuple, Dict[str, Any]] = serialized_field()
    flat_peaks_df: pd.DataFrame = serialized_field()
    filtered_flat_peaks_df: pd.DataFrame = serialized_field()
    peak_counts: PeakCounts = serialized_field()

    @function_attributes(short_name=None, tags=['efficiency', 'contours'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-06 15:54', related_items=[])
    @classmethod
    def perform_convert_paths_to_vertices(cls, result_obj: "PosteriorPeaksPeakProminence2dResult", in_place: bool = False) -> "PosteriorPeaksPeakProminence2dResult":
        """Convert matplotlib Path objects in level_slices to vertex arrays.
        
        Parameters:
        -----------
        result_obj : PosteriorPeaksPeakProminence2dResult
            The result object to convert
        in_place : bool
            If True, modifies the object in place. If False, returns a new object.
            
        Returns:
        --------
        PosteriorPeaksPeakProminence2dResult
            Converted object (same object if in_place=True)
        """
        from matplotlib.path import Path
        
        if not in_place:
            # Create a copy by reconstructing from state
            result_obj = cls._reload_class(result_obj)
        
        def convert_path_to_vertices(path_obj):
            """Helper to convert a single Path object to vertices."""
            if isinstance(path_obj, Path):
                return path_obj.vertices.copy()
            elif isinstance(path_obj, np.ndarray):
                # Already converted, return as is
                return path_obj
            else:
                return path_obj  # Unknown type, return as is
        
        # Traverse results dictionary
        for result_key, slab_result_dict in result_obj.results.items():
            peaks_dict = slab_result_dict.get('peaks', {})
            
            for peak_id, peak_info in peaks_dict.items():
                # Convert top-level 'contour' if present
                if 'contour' in peak_info:
                    peak_info['contour'] = convert_path_to_vertices(peak_info['contour'])
                
                # Convert top-level 'contours' list if present
                if 'contours' in peak_info:
                    contours_list = peak_info['contours']
                    if isinstance(contours_list, list):
                        peak_info['contours'] = [convert_path_to_vertices(contour) for contour in contours_list]
                
                # Convert 'contour' in level_slices
                level_slices = peak_info.get('level_slices', {})
                for probe_level, slice_info in level_slices.items():
                    if 'contour' in slice_info:
                        slice_info['contour'] = convert_path_to_vertices(slice_info['contour'])
                    
                    # Also check for 'contours' in level_slices (if it exists)
                    if 'contours' in slice_info:
                        contours_list = slice_info['contours']
                        if isinstance(contours_list, list):
                            slice_info['contours'] = [convert_path_to_vertices(contour) for contour in contours_list]
        
        return result_obj


    def convert_paths_to_vertices(self, in_place: bool = False) -> "PosteriorPeaksPeakProminence2dResult":
        """ if inplace == False it returns a copy of self. """
        return self.perform_convert_paths_to_vertices(result_obj=self, in_place=in_place)
    


    # For serialization/pickling: ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        # # Remove the unpicklable entries.
        # _non_pickled_fields = ['curr_active_pipeline', 'track_templates']
        # for a_non_pickleable_field in _non_pickled_fields:
        #     del state[a_non_pickleable_field]
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        # For `VersionedResultMixin`
        self._VersionedResultMixin__setstate__(state)

        # _non_pickled_field_restore_defaults = dict(zip(['curr_active_pipeline', 'track_templates'], [None, None]))
        # for a_field_name, a_default_restore_value in _non_pickled_field_restore_defaults.items():
        #     if a_field_name not in state:
        #         state[a_field_name] = a_default_restore_value

        self.__dict__.update(state)
        # # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        # super(PosteriorPeaksPeakProminence2dResult, self).__init__() # from

    def __repr__(self):
        """ 2024-01-11 - Renders only the fields and their sizes
        """
        from pyphocorehelpers.print_helpers import strip_type_str_to_classname
        attr_reprs = []
        for a in self.__attrs_attrs__:
            attr_type = strip_type_str_to_classname(type(getattr(self, a.name)))
            if 'shape' in a.metadata:
                shape = ', '.join(a.metadata['shape'])  # this joins tuple elements with a comma, creating a string without quotes
                attr_reprs.append(f"{a.name}: {attr_type} | shape ({shape})")  # enclose the shape string with parentheses
            else:
                attr_reprs.append(f"{a.name}: {attr_type}")
        content = ",\n\t".join(attr_reprs)
        return f"{type(self).__name__}({content}\n)"

    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, debug_print=False, enable_hdf_testing_mode:bool=False, OVERRIDE_ALLOW_GLOBAL_NESTED_EXPANSION:bool=None, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        # simplified_obj = PosteriorPeaksPeakProminence2dResult.convert_paths_to_vertices(result_obj=self, in_place=False)
        # simplified_obj
        super().to_hdf(file_path, key=key, debug_print=debug_print, enable_hdf_testing_mode=enable_hdf_testing_mode, OVERRIDE_ALLOW_GLOBAL_NESTED_EXPANSION=OVERRIDE_ALLOW_GLOBAL_NESTED_EXPANSION, **kwargs)


    @classmethod
    def _reload_class(cls, an_instance):
        """ specifically updates the instance after its class definition has been updated.
        """
        non_init_subset=['_VersionedResultMixin_version']

        _full_state = an_instance.__getstate__()
        _init_state = get_dict_subset(_full_state, subset_excludelist=non_init_subset)
        _post_init_state = get_dict_subset(_full_state, subset_includelist=non_init_subset)
        _obj = cls(**_init_state)
        _obj.__dict__.update(**_post_init_state) ## perform literal update
        return _obj


@function_attributes(short_name=None, tags=['internal', 'parallelizable'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-06 06:48', related_items=[])
def _compute_single_posterior_slab(epoch_idx: int, t_idx: int, slab: NDArray, xbin_centers: NDArray, ybin_centers: NDArray, step: float, min_considered_promenence: float, peak_height_multiplier_probe_levels: Tuple, debug_print: bool = False, should_return_raw_matplotlib_Path_contours: bool=False):
    """Internal helper: compute peak prominence results for a single (epoch, t_idx) slab.

    Returns:
        (epoch_idx, t_idx, posterior_peaks_df, slab_result_dict)
    """
    def _subfn_get_contour_curve(contour):
        """ captures should_return_raw_matplotlib_Path_contours """
        if contour is None:
            return contour
        if should_return_raw_matplotlib_Path_contours:
            return contour
        else:
            return contour.vertices.copy(),  # contour_verticies Store as numpy array (N, 2)


    # This mirrors the inner loop logic that previously lived inside
    # PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation.
    _, _, slab, peaks_dict, id_map, prominence_map, parent_map = compute_prominence_contours(
        xbin_centers=xbin_centers,
        ybin_centers=ybin_centers,
        slab=slab,
        step=step,
        min_area=None,
        min_considered_promenence=min_considered_promenence,
        include_edge=True,
        verbose=False
    )

    n_peaks = len(peaks_dict)
    if n_peaks == 0:
        # Nothing to record for this (epoch, t_idx)
        empty_df = pd.DataFrame()
        slab_result_dict = {
            'peaks': peaks_dict,
            'slab': slab,
            'id_map': id_map,
            'prominence_map': prominence_map,
            'parent_map': parent_map
        }
        return epoch_idx, t_idx, empty_df, slab_result_dict

    n_slices = len(peak_height_multiplier_probe_levels)

    # arrays sized per peak and per slice level
    n_total_cell_slice_results = n_slices * n_peaks

    # Peak
    summit_slice_peak_id_arr = np.zeros((n_peaks, n_slices), dtype=np.int16)
    summit_slice_peak_level_multiplier_arr = np.zeros((n_peaks, n_slices), dtype=float)
    summit_slice_peak_level_arr = np.zeros((n_peaks, n_slices), dtype=float)
    summit_slice_peak_height_arr = np.zeros((n_peaks, n_slices), dtype=float)
    summit_slice_peak_prominence_arr = np.zeros((n_peaks, n_slices), dtype=float)
    summit_peak_center_x_arr = np.zeros((n_peaks, n_slices), dtype=float)
    summit_peak_center_y_arr = np.zeros((n_peaks, n_slices), dtype=float)

    # Slice geometry
    summit_slice_idx_arr = np.tile(np.arange(n_slices), n_peaks).astype('int')
    summit_slice_x_side_length_arr = np.zeros((n_peaks, n_slices), dtype=float)
    summit_slice_y_side_length_arr = np.zeros((n_peaks, n_slices), dtype=float)
    summit_slice_center_x_arr = np.zeros((n_peaks, n_slices), dtype=float)
    summit_slice_center_y_arr = np.zeros((n_peaks, n_slices), dtype=float)

    for peak_idx, (peak_id, a_peak_dict) in enumerate(peaks_dict.items()):
        if debug_print:
            print(f'computing contours for epoch[{epoch_idx}], t[{t_idx}], peak_id: {peak_id}...')

        summit_slice_peak_height_arr[peak_idx, :] = a_peak_dict['height']
        summit_slice_peak_prominence_arr[peak_idx, :] = a_peak_dict['prominence']
        summit_peak_center_x_arr[peak_idx, :] = a_peak_dict['center'][0]
        summit_peak_center_y_arr[peak_idx, :] = a_peak_dict['center'][1]

        # probe levels for this peak
        a_peak_dict['probe_levels'] = np.array(
            [a_peak_dict['height'] * multiplier for multiplier in peak_height_multiplier_probe_levels],
            dtype=float
        )
        summit_slice_peak_level_multiplier_arr[peak_idx, :] = np.array(
            peak_height_multiplier_probe_levels, dtype=float
        )
        summit_slice_peak_level_arr[peak_idx, :] = a_peak_dict['probe_levels']

        included_computed_contours = PeakPromenence._find_contours_at_levels(
            xbin_centers, ybin_centers, slab, a_peak_dict['center'], a_peak_dict['probe_levels']
        )

        # Build the dict that contains the output level slices
        a_peak_dict['level_slices'] = {
            probe_lvl: {
                # 'contour': contour,
                'contour': _subfn_get_contour_curve(contour=contour), # contour_verticies Store as numpy array (N, 2)
                'bbox': contour.get_extents(),
                'size': contour.get_extents().size
            }
            for probe_lvl, contour in included_computed_contours.items()
            if (contour is not None)
        }

        if debug_print:
            print(f"probe_levels: {a_peak_dict['probe_levels']}")

        # Build flat output:
        for lvl_idx, probe_lvl in enumerate(a_peak_dict['probe_levels']):
            a_slice = a_peak_dict['level_slices'].get(probe_lvl, None)
            if a_slice is None:
                print('WARNING: a_slice is None in posterior prominence computation; skipping this slice.')
            else:
                slice_bbox = a_slice['bbox']
                (x0, y0, width, height) = slice_bbox.bounds
                summit_slice_peak_id_arr[peak_idx, lvl_idx] = peak_id
                summit_slice_x_side_length_arr[peak_idx, lvl_idx] = width
                summit_slice_y_side_length_arr[peak_idx, lvl_idx] = height
                summit_slice_center_x_arr[peak_idx, lvl_idx] = float(x0) + (0.5 * float(width))
                summit_slice_center_y_arr[peak_idx, lvl_idx] = float(y0) + (0.5 * float(height))

    if debug_print:
        print(f'building peak_df for epoch[{epoch_idx}], t[{t_idx}] with {n_peaks} peaks...')

    # For posteriors, use a simple peak_height definition identical to peak_relative_height:
    peak_relative_height_flat = summit_slice_peak_height_arr.flatten()

    posterior_peaks_df = pd.DataFrame({
        'epoch_idx': np.full((n_total_cell_slice_results,), epoch_idx, dtype=int),
        'time_bin_idx': np.full((n_total_cell_slice_results,), t_idx, dtype=int),
        'summit_idx': summit_slice_peak_id_arr.flatten(),
        'summit_slice_idx': summit_slice_idx_arr.flatten(),
        'slice_level_multiplier': summit_slice_peak_level_multiplier_arr.flatten(),
        'summit_slice_level': summit_slice_peak_level_arr.flatten(),
        'peak_relative_height': peak_relative_height_flat,
        'peak_prominence': summit_slice_peak_prominence_arr.flatten(),
        'peak_center_x': summit_peak_center_x_arr.flatten(),
        'peak_center_y': summit_peak_center_y_arr.flatten(),
        'summit_slice_x_width': summit_slice_x_side_length_arr.flatten(),
        'summit_slice_y_width': summit_slice_y_side_length_arr.flatten(),
        'summit_slice_center_x': summit_slice_center_x_arr.flatten(),
        'summit_slice_center_y': summit_slice_center_y_arr.flatten()
    })
    posterior_peaks_df['peak_height'] = peak_relative_height_flat

    if debug_print:
        print('done building peak_df for posterior slab.')

    slab_result_dict = {
        'peaks': peaks_dict,
        'slab': slab,
        'id_map': id_map,
        'prominence_map': prominence_map,
        'parent_map': parent_map
    }
    return epoch_idx, t_idx, posterior_peaks_df, slab_result_dict



# ==================================================================================================================================================================================================================================================================================== #
# Main Computations                                                                                                                                                                                                                                                                    #
# ==================================================================================================================================================================================================================================================================================== #
@metadata_attributes(short_name=None, tags=['peak', 'promenence-2d', 'promenence', 'helper'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-12-21 00:00', related_items=[])
class PeakPromenence:
    """ 
        from pyphoplacecellanalysis.External.peak_prominence2d import PeakPromenence

    """

    @function_attributes(short_name=None, tags=['private'], input_requires=[], output_provides=[], uses=[], used_by=['_perform_find_posterior_peaks_peak_prominence2d_computation'], creation_date='2025-12-21 00:00', related_items=[])
    @classmethod
    def _find_contours_at_levels(cls, xbin_centers: NDArray, ybin_centers: NDArray, slab: NDArray, peak_probe_point: NDArray, probe_levels: NDArray):
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
        ax.set_axis_off()  # Disable axis rendering
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
        ## END for ii, levii in enumerate(probe_levels[::-1])...

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
        ## END for a_peak_row in filtered_flat_peaks_df[['peak_center_binned_x', 'peak_center_binned_y']].itertuples()...

        return peak_nearest_directional_boundary_bins, peak_nearest_directional_boundary_displacements, peak_nearest_directional_boundary_distances


    @function_attributes(short_name=None, tags=['peak', 'promenence-2d'], input_requires=[], output_provides=[], uses=['compute_prominence_contours', 'PeakPromenence._find_contours_at_levels'], used_by=[], creation_date='2025-12-21 00:00', related_items=['_perform_pf_find_ratemap_peaks_peak_prominence2d_computation', 'compute_2d_peak_prominence'])
    @classmethod
    def _perform_find_posterior_peaks_peak_prominence2d_computation(cls, p_x_given_n_list: List[NDArray], xbin_centers: NDArray, ybin_centers: NDArray, step: float = 0.01, peak_height_multiplier_probe_levels: Tuple = (0.5, 0.9), minimum_included_peak_height: float = 0.2, min_considered_promenence: float = 0.2, uniform_blur_size: int = 3, gaussian_blur_sigma: float = 3, debug_print: bool = False, parallel: bool = True, max_workers: Optional[int] = None) -> 'PosteriorPeaksPeakProminence2dResult':
            """Uses the peak_prominence2d package to find the peaks and prominences of 2D decoded posteriors.

            History:
                Based off of '_perform_pf_find_ratemap_peaks_peak_prominence2d_computation'

            This is analogous to `_perform_pf_find_ratemap_peaks_peak_prominence2d_computation`, but operates on
            decoded posterior probability distributions instead of placefield tuning curves.

            Inputs:
                p_x_given_n_list: list of per-epoch posterior arrays. Each element should be either:
                    - 3D: (n_xbins, n_ybins, n_time_bins) or
                    - 2D: (n_xbins, n_ybins) for single-time-bin epochs.
                xbin_centers, ybin_centers: spatial bin centers for the posterior grid.
                step: float, contour interval used for prominence calculation. Finer (smaller) values give better accuracy but slower computation. Default: 0.01.
                peak_height_multiplier_probe_levels: slice levels as fractions of peak height (e.g., (0.5, 0.9)).
                minimum_included_peak_height: threshold applied to the `peak_height` column when filtering.
                uniform_blur_size: int, size parameter for uniform filter applied to peak counts map.
                    Default: 3.
                gaussian_blur_sigma: float, sigma parameter for Gaussian filter applied to peak counts map.
                    Default: 3.0.
                debug_print: bool, if True, print verbose debugging information during computation.
                    Default: False.
                parallel: bool, if True process (epoch_idx, t_idx) slabs in parallel using a process pool.
                    When False, behavior is identical to the original serial implementation.
                max_workers: Optional[int], maximum number of worker processes for parallel mode. If None,
                    the default from ProcessPoolExecutor is used.

            Returns:
                PosteriorPeaksPeakProminence2dResult with:
                    xx, yy: xbin_centers, ybin_centers
                    results: dict keyed by (epoch_idx, time_bin_idx) with per-slab peak results:
                        {'peaks': peaks_dict, 'slab': slab, 'id_map': id_map,
                         'prominence_map': prominence_map, 'parent_map': parent_map}
                    flat_peaks_df: concatenated DataFrame of all peaks across epochs/time-bins
                    filtered_flat_peaks_df: filtered subset used for peak-count maps
                    peak_counts: PeakCounts(raw=..., uniform_blurred=..., gaussian_blurred=...)

            Usage (example):
                >>> decoded_epochs_result = some_decoder_result  # has .p_x_given_n_list
                >>> xbin_centers = decoded_epochs_result.xbin_centers
                >>> ybin_centers = decoded_epochs_result.ybin_centers
                >>> posterior_peaks = PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation(
                ...     decoded_epochs_result.p_x_given_n_list,
                ...     xbin_centers,
                ...     ybin_centers,
                ...     step=0.02,
                ...     peak_height_multiplier_probe_levels=(0.5, 0.9),
                ...     minimum_included_peak_height=0.2,
                ...     uniform_blur_size=3,
                ...     gaussian_blur_sigma=3,
                ...     debug_print=False)
                >>> posterior_peaks.flat_peaks_df.head()
            """
            from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns
            from scipy.ndimage.filters import uniform_filter, gaussian_filter

            n_epochs = len(p_x_given_n_list)

            # infer edges from centers for later binning (xbin, ybin)
            xbin_centers = np.asarray(xbin_centers)
            ybin_centers = np.asarray(ybin_centers)
            if xbin_centers.ndim != 1 or ybin_centers.ndim != 1:
                raise ValueError('xbin_centers and ybin_centers must be 1D arrays.')
            if len(xbin_centers) < 2 or len(ybin_centers) < 2:
                raise ValueError('xbin_centers and ybin_centers must each have length >= 2.')

            x_edges = np.concatenate(([xbin_centers[0] - (xbin_centers[1] - xbin_centers[0]) / 2.0],
                                      (xbin_centers[:-1] + xbin_centers[1:]) / 2.0,
                                      [xbin_centers[-1] + (xbin_centers[-1] - xbin_centers[-2]) / 2.0]))
            y_edges = np.concatenate(([ybin_centers[0] - (ybin_centers[1] - ybin_centers[0]) / 2.0],
                                      (ybin_centers[:-1] + ybin_centers[1:]) / 2.0,
                                      [ybin_centers[-1] + (ybin_centers[-1] - ybin_centers[-2]) / 2.0]))

            # Build list of slabs (epoch_idx, t_idx, slab) to process
            tasks = []
            for epoch_idx in np.arange(n_epochs):
                p_x_given_n = np.asarray(p_x_given_n_list[epoch_idx])

                if p_x_given_n.ndim == 2:
                    # (n_xbins, n_ybins) => add singleton time dimension
                    p_x_given_n = p_x_given_n[:, :, np.newaxis]
                elif p_x_given_n.ndim != 3:
                    raise ValueError(f'p_x_given_n for epoch {epoch_idx} must be 2D or 3D, got shape {p_x_given_n.shape}')

                n_xbins, n_ybins, n_time_bins = p_x_given_n.shape
                if (n_xbins != len(xbin_centers)) or (n_ybins != len(ybin_centers)):
                    raise ValueError(f'epoch {epoch_idx}: posterior shape {(n_xbins, n_ybins)} does not match x/y bin centers '
                                     f'({len(xbin_centers)}, {len(ybin_centers)})')

                for t_idx in np.arange(n_time_bins):
                    a_p_x_given_n = np.squeeze(p_x_given_n[:, :, t_idx])
                    slab = a_p_x_given_n.T  # match compute_prominence_contours convention
                    tasks.append((epoch_idx, t_idx, slab))

            out_results = {}
            out_posteriors_peak_dfs_list = []

            # Decide whether to run in parallel or serial
            n_tasks = len(tasks)
            n_cpus = os.cpu_count() or 1
            use_parallel = parallel and (n_tasks > 1) and (n_cpus > 1)

            if use_parallel:
                # Process in parallel using a process pool
                futures = []
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    for epoch_idx, t_idx, slab in tasks:
                        fut = ex.submit(
                            _compute_single_posterior_slab,
                            epoch_idx, t_idx, slab,
                            xbin_centers, ybin_centers,
                            step, min_considered_promenence,
                            peak_height_multiplier_probe_levels,
                            debug_print
                        )
                        futures.append(fut)

                    results_list = [fut.result() for fut in as_completed(futures)]
            else:
                # Serial fallback: identical behavior to original implementation, but via helper
                results_list = []
                for epoch_idx, t_idx, slab in tasks:
                    result = _compute_single_posterior_slab(
                        epoch_idx, t_idx, slab,
                        xbin_centers, ybin_centers,
                        step, min_considered_promenence,
                        peak_height_multiplier_probe_levels,
                        debug_print
                    )
                    results_list.append(result)

            # Reconstruct in deterministic order: sort by (epoch_idx, t_idx)
            results_list.sort(key=lambda tup: (tup[0], tup[1]))

            for epoch_idx, t_idx, posterior_peaks_df, slab_result_dict in results_list:
                out_results[(epoch_idx, t_idx)] = slab_result_dict
                if not posterior_peaks_df.empty:
                    out_posteriors_peak_dfs_list.append(posterior_peaks_df)

            if len(out_posteriors_peak_dfs_list) == 0:
                # no peaks found anywhere; return empty structures
                empty_df = pd.DataFrame()
                empty_counts = np.zeros((len(xbin_centers), len(ybin_centers)), dtype=int)
                empty_counts_blurred = uniform_filter(empty_counts.astype('float'), size=uniform_blur_size, mode='constant')
                empty_counts_blurred_gaussian = gaussian_filter(empty_counts.astype('float'), sigma=gaussian_blur_sigma)
                peak_counts_results = PeakCounts(raw=empty_counts, uniform_blurred=empty_counts_blurred, gaussian_blurred=empty_counts_blurred_gaussian)
                return PosteriorPeaksPeakProminence2dResult(xx=xbin_centers, yy=ybin_centers, results=out_results, flat_peaks_df=empty_df, filtered_flat_peaks_df=empty_df, peak_counts=peak_counts_results)

            # Build final concatenated dataframe:
            if debug_print:
                print(f'building final concatenated posterior cell_peaks_df for {n_epochs} epochs...')
            posterior_peaks_df = pd.concat(out_posteriors_peak_dfs_list, ignore_index=True)

            # Find which position bin each peak falls in and add it to the flat_peaks_df:
            posterior_peaks_df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(
                posterior_peaks_df, bin_values=(x_edges, y_edges),
                position_column_names=('peak_center_x', 'peak_center_y'),
                binned_column_names=('peak_center_binned_x', 'peak_center_binned_y'),
                active_computation_config=None, force_recompute=False, debug_print=debug_print)

            # Filter the summits, compute peak-counts, etc:
            # We do not currently have an EloyAnalysis-like object for posteriors, so pass None.
            active_eloy_analysis = None
            filtered_summits_analysis_df, pf_peak_counts_map = PeakPromenence._build_filtered_summits_analysis_results(
                xbin, ybin, np.arange(1, len(xbin)), np.arange(1, len(ybin)),
                posterior_peaks_df, active_eloy_analysis,
                slice_level_multiplier=0.5,
                minimum_included_peak_height=minimum_included_peak_height,
                debug_print=debug_print)

            pf_peak_counts_map_blurred = uniform_filter(pf_peak_counts_map.astype('float'), size=uniform_blur_size, mode='constant')
            pf_peak_counts_map_blurred_gaussian = gaussian_filter(pf_peak_counts_map.astype('float'), sigma=gaussian_blur_sigma)
            pf_peak_counts_results = PeakCounts(raw=pf_peak_counts_map,
                                                       uniform_blurred=pf_peak_counts_map_blurred,
                                                       gaussian_blurred=pf_peak_counts_map_blurred_gaussian)

            return PosteriorPeaksPeakProminence2dResult(xx=xbin_centers, yy=ybin_centers, results=out_results,
                                     flat_peaks_df=posterior_peaks_df,
                                     filtered_flat_peaks_df=filtered_summits_analysis_df,
                                     peak_counts=pf_peak_counts_results)






    # ==================================================================================================================================================================================================================================================================================== #
    # 2025-12-22 - New High-efficiency 2D peak promenence calculations                                                                                                                                                                                                                     #
    # ==================================================================================================================================================================================================================================================================================== #

    @function_attributes(short_name=None, tags=['high-efficiency', 'rewrite'], input_requires=[], output_provides=[], uses=[], used_by=['cls.compute_2d_dt_posterior_peak_promenences'], creation_date='2025-12-23 08:44', related_items=['_perform_find_posterior_peaks_peak_prominence2d_computation'])
    @classmethod
    def compute_2d_peak_prominence(cls, Z_2d: NDArray[ND.Shape["N_XBINS, N_YBINS"], Any]):
        """
        Computes prominence for all 2D local maxima in Z_2d.

        High-efficiency rewrite of `_perform_find_posterior_peaks_peak_prominence2d_computation`


        Returns:
            peak_coords: (N, 2) array of (x, y) peak locations
            prominences: (N,) prominence values
        """
        if Z_2d.ndim != 2:
            raise ValueError(f"compute_2d_peak_prominence expects a 2D array, got shape {Z_2d.shape}")

        # --- find local maxima ---
        neighborhood = ndimage.generate_binary_structure(2, 2)
        local_max = (Z_2d == ndimage.maximum_filter(Z_2d, footprint=neighborhood))
        local_max &= (Z_2d > np.min(Z_2d))

        peak_coords = np.column_stack(np.nonzero(local_max))
        peak_heights = Z_2d[local_max]

        # --- compute prominence surface ---
        seed = Z_2d.copy()
        seed[local_max] = -np.inf

        reconstructed = reconstruction(seed, Z_2d, method="dilation")

        prominences = peak_heights - reconstructed[local_max]

        return peak_coords, prominences


    @function_attributes(short_name=None, tags=['high-efficiency', 'rewrite'], input_requires=[], output_provides=[], uses=['cls.compute_2d_peak_prominence'], used_by=['cls.compute_posterior_peak_promenences'], creation_date='2025-12-23 08:44', related_items=[])
    @classmethod
    def compute_2d_dt_posterior_peak_promenences(cls, a_p_x_given_n: NDArray[ND.Shape["N_XBINS, N_YBINS, N_TBINS"], Any], alpha: Union[float, List[float]] = 0.9):
        """ for a single posterior (from a single decoded epoch, etc) process each time bin

        epoch_promenences, epoch_masks = PeakPromenence.compute_2d_posterior_peak_promenences(a_p_x_given_n=a_p_x_given_n, alpha=alpha)

                [peak_heights[np.argmax(peak_heights)] for (peak_coords, prominences, peak_heights) in epoch_promenence_tuples]
                
                all_t_bin_peak_heights: NDArray = np.array([np.nanmax(peak_heights) for (peak_coords, prominences, peak_heights) in epoch_promenence_tuples])
                

        """
        def _subfn_compute_promenence_alpha_level(Z_2d, peak_coords, peak_heights, a_peak_idx: int, an_alpha: float):

            px, py = peak_coords[a_peak_idx]
            peak_height_max: float = peak_heights[a_peak_idx]

            # --- threshold ---
            threshold_mask = Z_2d >= (an_alpha * peak_height_max)

            # --- connected components ---
            labeled, num = ndimage.label(threshold_mask)
            a_dominant_label = labeled[px, py]
            a_dominant_peak_mask = (labeled == a_dominant_label)
            return a_dominant_label, a_dominant_peak_mask

        # ==================================================================================================================================================================================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
        # ==================================================================================================================================================================================================================================================================================== #
        if a_p_x_given_n.ndim != 3:
            raise ValueError(f"compute_2d_posterior_peak_promenences expects a 3D array, got shape {a_p_x_given_n.shape}")

        if np.isscalar(alpha):
            alpha = [alpha] ## make into a single element list

        n_t_bins = np.shape(a_p_x_given_n)[-1]
        
        epoch_promenence_tuples: List[Tuple] = []
        epoch_masks: List[List[NDArray]] = []
        # epoch_masks_dict Dict[float, List[List[NDArray]]] = {an_alpha:[] for an_alpha in alpha}
        
        for t_idx in range(n_t_bins):
            Z_2d = a_p_x_given_n[:, :, t_idx]
            peak_coords, prominences = cls.compute_2d_peak_prominence(Z_2d=Z_2d)

            # if no peaks were found for this time bin, record an all-False mask and continue
            if peak_coords.size == 0:
                # Create a list of masks (one for each alpha), all False
                dominant_peak_mask = [np.zeros_like(Z_2d, dtype=bool) for _ in alpha]
                epoch_masks.append(dominant_peak_mask)
                epoch_promenence_tuples.append((peak_coords, prominences, np.array([])))
                continue
            
            # --- identify dominant peak ---
            peak_heights = Z_2d[peak_coords[:, 0], peak_coords[:, 1]]
            dominant_peak_idx: int = np.argmax(peak_heights)

            dominant_peak_mask = []
            for an_alpha in alpha:
                a_dominant_label, a_dominant_peak_mask = _subfn_compute_promenence_alpha_level(Z_2d=Z_2d, peak_coords=peak_coords, peak_heights=peak_heights, a_peak_idx=dominant_peak_idx, an_alpha=an_alpha)
                dominant_peak_mask.append(a_dominant_peak_mask)
                # epoch_masks_dict[an_alpha].append(a_dominant_peak_mask)
            
            ## OUTPUTS: dominant_peak_mask
            epoch_masks.append(dominant_peak_mask)
            epoch_promenence_tuples.append((peak_coords, prominences, peak_heights))

        ## END for t_idx in range(n_t_bins)...
        epoch_masks: List[NDArray] = [np.stack([a_mask[an_alpha_idx] for a_t_idx, a_mask in enumerate(epoch_masks)], axis=-1) for an_alpha_idx, an_alpha in enumerate(alpha)] # ValueError: all input arrays must have the same shape
        
        # try:
            # for an_alpha, v in epoch_masks_dict.items():
            #     try:
            #         epoch_masks_dict[an_alpha] = np.stack(v, axis=-1)
            #     except Exception as e:
            #         raise e

            # epoch_masks_dict = {an_alpha:np.stack(v, axis=-1) for an_alpha, v in epoch_masks_dict.items()}
        # except Exception as e:
        #     raise e

        # return epoch_promenence_tuples, epoch_masks_dict
        return epoch_promenence_tuples, epoch_masks


    @function_attributes(short_name=None, tags=['high-efficiency', 'rewrite'], input_requires=[], output_provides=[], uses=['cls.compute_2d_dt_posterior_peak_promenences'], used_by=[], creation_date='2025-12-23 08:44', related_items=[])
    @classmethod
    def compute_posterior_peak_promenences(cls, p_x_given_n_list: List[NDArray[ND.Shape["N_XBINS, N_YBINS, N_TBINS"], Any]], alpha: Union[float, List[float]] = 0.9):
        """ 
        from pyphoplacecellanalysis.External.peak_prominence2d import PeakPromenence

        all_epochs_all_t_bins_epoch_t_bin_idx_tuple_list, all_epochs_promenence_tuples_dict, all_epochs_masks = PeakPromenence.compute_posterior_peak_promenences(p_x_given_n_list=a_widget.decoded_result.p_x_given_n_list, alpha=0.9)
        
        """
        all_epochs_all_t_bins_epoch_t_bin_idx_tuple_list: List[DecodedEpochTimeBinIndexTuple] = []
        all_epochs_promenence_tuples_dict: Dict[DecodedEpochTimeBinIndexTuple, Tuple] = {}
        all_epochs_promenence_tuples: List[Tuple] = []
        all_epochs_masks: List[List[NDArray]] = []
        
        n_epochs: int = len(p_x_given_n_list)

        for epoch_idx, a_p_x_given_n in enumerate(p_x_given_n_list):
            n_t_bins = np.shape(a_p_x_given_n)[-1]
            epoch_promenences, epoch_masks_dict = cls.compute_2d_dt_posterior_peak_promenences(a_p_x_given_n=a_p_x_given_n, alpha=alpha)
            all_epochs_promenence_tuples.append(epoch_promenences)

            curr_epoch_t_bin_idx_tuple = (epoch_idx, None)
            ## create the dict index by the (epoch_idx, epoch_t_bin_idx) tuple:
            all_epochs_all_t_bins_epoch_t_bin_idx_tuple_list.extend([(epoch_idx, a_t_bin_idx) for a_t_bin_idx in np.arange(n_t_bins)]) ## should the t-bins be 1-index? Nah I don't thinks o.
            curr_epoch_promp_tuples_dict = {(epoch_idx, a_t_bin_idx):a_prom_tuple for a_t_bin_idx, a_prom_tuple in enumerate(epoch_promenences)} ## each value of a_prom_tuple is (peak_coords, prominences, peak_heights)
            all_epochs_promenence_tuples_dict.update(curr_epoch_promp_tuples_dict)

            # epoch_masks: List[NDArray] = [np.stack(an_alpha_epoch_masks, axis=-1) for an_alpha_epoch_masks in epoch_masks_list] # List[(41, 63, 5)] - List[(n_x_bins, n_y_bins, n_t_bins)] (one for each value of alpha)
            # epoch_masks: List[List[NDArray]] = epoch_masks_list
            # assert np.shape(epoch_masks) == np.shape(a_p_x_given_n)
            # all_epochs_masks.append(epoch_masks)
            all_epochs_masks.append(epoch_masks_dict)
            
        ## END for i, a_p_x_given_n in enumerat

        return all_epochs_all_t_bins_epoch_t_bin_idx_tuple_list, all_epochs_promenence_tuples_dict, all_epochs_masks



    # ==================================================================================================================================================================================================================================================================================== #
    # 2026-01-06 - Compatibility for returned results of `compute_posterior_peak_promenences` with outputs of `_perform_find_posterior_peaks_peak_prominence2d_computation`                                                                                                                #
    # ==================================================================================================================================================================================================================================================================================== #
    @function_attributes(short_name=None, tags=['COMPATIBILITY', 'UNFINISHED', 'UNTESTED'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-05 09:14', related_items=[])
    @classmethod
    def _reconstruct_posterior_peaks_from_efficient_computation(cls, p_x_given_n_list: List[NDArray], xbin_centers: NDArray, ybin_centers: NDArray, all_epochs_promenence_tuples: List[List[Tuple]], peak_height_multiplier_probe_levels: Tuple = (0.5, 0.9), minimum_included_peak_height: float = 0.2, uniform_blur_size: int = 3, gaussian_blur_sigma: float = 3, debug_print: bool = False) -> 'DynamicParameters':
        """Reconstructs the same output structure as `_perform_find_posterior_peaks_peak_prominence2d_computation` 
        from the efficient outputs of `compute_posterior_peak_promenences`.
        
        This function takes the fast peak detection results and reconstructs the detailed peak dictionaries,
        DataFrames, and count maps without needing to run the slower `compute_prominence_contours` computation.
        
        Note: This does NOT include id_map, prominence_map, parent_map, or parent relationships, as these
        require the full getProminence computation. These are set to None in the results.
        
        Inputs:
            p_x_given_n_list: list of per-epoch posterior arrays. Each element should be either:
                - 3D: (n_xbins, n_ybins, n_time_bins) or
                - 2D: (n_xbins, n_ybins) for single-time-bin epochs.
            xbin_centers, ybin_centers: spatial bin centers for the posterior grid.
            all_epochs_promenence_tuples: output from `compute_posterior_peak_promenences`, 
                List[List[Tuple]] where each inner list contains tuples of (peak_coords, prominences, peak_heights)
                for each time bin in that epoch. peak_coords is (N, 2) with [x_idx, y_idx] in original array coordinates.
            peak_height_multiplier_probe_levels: slice levels as fractions of peak height (e.g., (0.5, 0.9)).
            minimum_included_peak_height: threshold applied to the `peak_height` column when filtering.
            uniform_blur_size: int, size parameter for uniform filter applied to peak counts map.
            gaussian_blur_sigma: float, sigma parameter for Gaussian filter applied to peak counts map.
            debug_print: bool, if True, print verbose debugging information during computation.
        
        Returns:
            DynamicParameters with fields:
                xx, yy: xbin_centers, ybin_centers
                results: dict keyed by (epoch_idx, time_bin_idx) with per-slab peak results:
                    {'peaks': peaks_dict, 'slab': slab, 'id_map': None,
                    'prominence_map': None, 'parent_map': None}
                flat_peaks_df: concatenated DataFrame of all peaks across epochs/time-bins
                filtered_flat_peaks_df: filtered subset used for peak-count maps
                peak_counts: DynamicParameters(raw=..., uniform_blurred=..., gaussian_blurred=...)
        """
        from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns
        from scipy.ndimage.filters import uniform_filter, gaussian_filter
        import pandas as pd
        
        n_epochs = len(p_x_given_n_list)
        n_slices = len(peak_height_multiplier_probe_levels)
        
        # Infer edges from centers for later binning (xbin, ybin)
        xbin_centers = np.asarray(xbin_centers)
        ybin_centers = np.asarray(ybin_centers)
        if xbin_centers.ndim != 1 or ybin_centers.ndim != 1:
            raise ValueError('xbin_centers and ybin_centers must be 1D arrays.')
        if len(xbin_centers) < 2 or len(ybin_centers) < 2:
            raise ValueError('xbin_centers and ybin_centers must each have length >= 2.')
        
        x_edges = np.concatenate(([xbin_centers[0] - (xbin_centers[1] - xbin_centers[0]) / 2.0],
                                (xbin_centers[:-1] + xbin_centers[1:]) / 2.0,
                                [xbin_centers[-1] + (xbin_centers[-1] - xbin_centers[-2]) / 2.0]))
        y_edges = np.concatenate(([ybin_centers[0] - (ybin_centers[1] - ybin_centers[0]) / 2.0],
                                (ybin_centers[:-1] + ybin_centers[1:]) / 2.0,
                                [ybin_centers[-1] + (ybin_centers[-1] - ybin_centers[-2]) / 2.0]))
        
        # Build the results:
        out_results = {}
        out_posteriors_peak_dfs_list = []
        
        for epoch_idx in np.arange(n_epochs):
            p_x_given_n = np.asarray(p_x_given_n_list[epoch_idx])
            if p_x_given_n.ndim == 2:
                # (n_xbins, n_ybins) => add singleton time dimension
                p_x_given_n = p_x_given_n[:, :, np.newaxis]
            elif p_x_given_n.ndim != 3:
                raise ValueError(f'p_x_given_n for epoch {epoch_idx} must be 2D or 3D, got shape {p_x_given_n.shape}')
            
            n_xbins, n_ybins, n_time_bins = p_x_given_n.shape
            if (n_xbins != len(xbin_centers)) or (n_ybins != len(ybin_centers)):
                raise ValueError(f'epoch {epoch_idx}: posterior shape {(n_xbins, n_ybins)} does not match x/y bin centers '
                                f'({len(xbin_centers)}, {len(ybin_centers)})')
            

            epoch_promenence_tuples = all_epochs_promenence_tuples[epoch_idx]
            peak_coords, prominences, peak_heights = epoch_promenence_tuples

            for t_idx in np.arange(n_time_bins):
                a_p_x_given_n = np.squeeze(p_x_given_n[:, :, t_idx])
                slab = a_p_x_given_n.T  # match compute_prominence_contours convention
                
                # Get peak data from efficient computation
                # epoch_tbin_index_tuple = (epoch_idx, t_idx)
                # peak_coords, prominences, peak_heights = all_epochs_promenence_tuples[epoch_tbin_index_tuple]
                

                peak_coords, prominences, peak_heights = epoch_promenence_tuples[t_idx]
                
                n_peaks = len(peak_coords)
                if n_peaks == 0:
                    # nothing to record for this (epoch, t_idx)
                    continue
                
                # Build peaks_dict
                peaks_dict = {}
                n_total_cell_slice_results = n_slices * n_peaks
                
                # Arrays sized per peak and per slice level
                summit_slice_peak_id_arr = np.zeros((n_peaks, n_slices), dtype=np.int16)
                summit_slice_peak_level_multiplier_arr = np.zeros((n_peaks, n_slices), dtype=float)
                summit_slice_peak_level_arr = np.zeros((n_peaks, n_slices), dtype=float)
                summit_slice_peak_height_arr = np.zeros((n_peaks, n_slices), dtype=float)
                summit_slice_peak_prominence_arr = np.zeros((n_peaks, n_slices), dtype=float)
                summit_peak_center_x_arr = np.zeros((n_peaks, n_slices), dtype=float)
                summit_peak_center_y_arr = np.zeros((n_peaks, n_slices), dtype=float)
                summit_slice_idx_arr = np.tile(np.arange(n_slices), n_peaks).astype('int')
                summit_slice_x_side_length_arr = np.zeros((n_peaks, n_slices), dtype=float)
                summit_slice_y_side_length_arr = np.zeros((n_peaks, n_slices), dtype=float)
                summit_slice_center_x_arr = np.zeros((n_peaks, n_slices), dtype=float)
                summit_slice_center_y_arr = np.zeros((n_peaks, n_slices), dtype=float)
                
                for peak_idx in range(n_peaks):
                    # peak_coords is (N, 2) with [x_idx, y_idx] in original array coordinates
                    # (from np.nonzero on Z_2d which is (n_xbins, n_ybins))
                    x_idx = peak_coords[peak_idx, 0]
                    y_idx = peak_coords[peak_idx, 1]
                    peak_height = peak_heights[peak_idx]
                    prominence = prominences[peak_idx]
                    
                    # Convert to spatial coordinates
                    peak_center_x = xbin_centers[x_idx]
                    peak_center_y = ybin_centers[y_idx]
                    
                    # Create peak_id (1-indexed to match original convention)
                    peak_id = peak_idx + 1
                    
                    # Build peak dict similar to original structure
                    a_peak_dict = {
                        'height': peak_height,
                        'prominence': prominence,
                        'center': (peak_center_x, peak_center_y),
                        'parent': None,  # Not available from efficient computation
                        'contour': None,  # Not available from efficient computation
                    }
                    
                    # Compute probe levels
                    a_peak_dict['probe_levels'] = np.array(
                        [peak_height * multiplier for multiplier in peak_height_multiplier_probe_levels], dtype=float)
                    
                    # Find contours at probe levels
                    if debug_print:
                        print(f'computing contours for epoch[{epoch_idx}], t[{t_idx}], peak_id: {peak_id}...')
                    
                    included_computed_contours = cls._find_contours_at_levels(xbin_centers, ybin_centers, slab, a_peak_dict['center'], a_peak_dict['probe_levels'])
                    
                    # Build the dict that contains the output level slices
                    a_peak_dict['level_slices'] = {
                        probe_lvl: {'contour': contour,
                                    'bbox': contour.get_extents(),
                                    'size': contour.get_extents().size}
                        for probe_lvl, contour in included_computed_contours.items()
                        if (contour is not None)
                    }
                    
                    peaks_dict[peak_id] = a_peak_dict
                    
                    # Fill arrays for DataFrame
                    summit_slice_peak_height_arr[peak_idx, :] = peak_height
                    summit_slice_peak_prominence_arr[peak_idx, :] = prominence
                    summit_peak_center_x_arr[peak_idx, :] = peak_center_x
                    summit_peak_center_y_arr[peak_idx, :] = peak_center_y
                    summit_slice_peak_level_multiplier_arr[peak_idx, :] = np.array(peak_height_multiplier_probe_levels, dtype=float)
                    summit_slice_peak_level_arr[peak_idx, :] = a_peak_dict['probe_levels']
                    
                    # Build flat output:
                    for lvl_idx, probe_lvl in enumerate(a_peak_dict['probe_levels']):
                        a_slice = a_peak_dict['level_slices'].get(probe_lvl, None)
                        if a_slice is None:
                            if debug_print:
                                print(f'WARNING: a_slice is None for peak {peak_id}, level {probe_lvl}; skipping this slice.')
                            # Fill with NaN or zeros for missing slices
                            summit_slice_peak_id_arr[peak_idx, lvl_idx] = peak_id
                            summit_slice_x_side_length_arr[peak_idx, lvl_idx] = np.nan
                            summit_slice_y_side_length_arr[peak_idx, lvl_idx] = np.nan
                            summit_slice_center_x_arr[peak_idx, lvl_idx] = np.nan
                            summit_slice_center_y_arr[peak_idx, lvl_idx] = np.nan
                        else:
                            slice_bbox = a_slice['bbox']
                            (x0, y0, width, height) = slice_bbox.bounds
                            summit_slice_peak_id_arr[peak_idx, lvl_idx] = peak_id
                            summit_slice_x_side_length_arr[peak_idx, lvl_idx] = width
                            summit_slice_y_side_length_arr[peak_idx, lvl_idx] = height
                            summit_slice_center_x_arr[peak_idx, lvl_idx] = float(x0) + (0.5 * float(width))
                            summit_slice_center_y_arr[peak_idx, lvl_idx] = float(y0) + (0.5 * float(height))
                ## END for peak_idx in range(n_peaks)...
                
                if debug_print:
                    print(f'building peak_df for epoch[{epoch_idx}], t[{t_idx}] with {n_peaks} peaks...')
                
                # For posteriors, use a simple peak_height definition identical to peak_relative_height:
                peak_relative_height_flat = summit_slice_peak_height_arr.flatten()
                
                posterior_peaks_df = pd.DataFrame({
                    'epoch_idx': np.full((n_total_cell_slice_results,), epoch_idx, dtype=int),
                    'time_bin_idx': np.full((n_total_cell_slice_results,), t_idx, dtype=int),
                    'summit_idx': summit_slice_peak_id_arr.flatten(),
                    'summit_slice_idx': summit_slice_idx_arr.flatten(),
                    'slice_level_multiplier': summit_slice_peak_level_multiplier_arr.flatten(),
                    'summit_slice_level': summit_slice_peak_level_arr.flatten(),
                    'peak_relative_height': peak_relative_height_flat,
                    'peak_prominence': summit_slice_peak_prominence_arr.flatten(),
                    'peak_center_x': summit_peak_center_x_arr.flatten(),
                    'peak_center_y': summit_peak_center_y_arr.flatten(),
                    'summit_slice_x_width': summit_slice_x_side_length_arr.flatten(),
                    'summit_slice_y_width': summit_slice_y_side_length_arr.flatten(),
                    'summit_slice_center_x': summit_slice_center_x_arr.flatten(),
                    'summit_slice_center_y': summit_slice_center_y_arr.flatten()
                })
                posterior_peaks_df['peak_height'] = peak_relative_height_flat
                
                out_posteriors_peak_dfs_list.append(posterior_peaks_df)
                
                if debug_print:
                    print('done building peak_df for posterior slab.')
                
                out_results[(epoch_idx, t_idx)] = {
                    'peaks': peaks_dict,
                    'slab': slab,
                    'id_map': None,  # Not available from efficient computation
                    'prominence_map': None,  # Not available from efficient computation
                    'parent_map': None  # Not available from efficient computation
                }

                ## END for t_idx in np.arange(n_time_bins)...

        ## END for epoch_idx in np.arange(n_epochs)...
        
        if len(out_posteriors_peak_dfs_list) == 0:
            # no peaks found anywhere; return empty structures
            empty_df = pd.DataFrame()
            empty_counts = np.zeros((len(xbin_centers), len(ybin_centers)), dtype=int)
            empty_counts_blurred = uniform_filter(empty_counts.astype('float'), size=uniform_blur_size, mode='constant')
            empty_counts_blurred_gaussian = gaussian_filter(empty_counts.astype('float'), sigma=gaussian_blur_sigma)
            peak_counts_results = DynamicParameters(raw=empty_counts, uniform_blurred=empty_counts_blurred, gaussian_blurred=empty_counts_blurred_gaussian)
            return DynamicParameters(xx=xbin_centers, yy=ybin_centers, results=out_results, flat_peaks_df=empty_df, filtered_flat_peaks_df=empty_df, peak_counts=peak_counts_results)
        
        # Build final concatenated dataframe:
        if debug_print:
            print(f'building final concatenated posterior cell_peaks_df for {n_epochs} epochs...')
        posterior_peaks_df = pd.concat(out_posteriors_peak_dfs_list, ignore_index=True)
        
        # Find which position bin each peak falls in and add it to the flat_peaks_df:
        posterior_peaks_df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(
            posterior_peaks_df, bin_values=(x_edges, y_edges),
            position_column_names=('peak_center_x', 'peak_center_y'),
            binned_column_names=('peak_center_binned_x', 'peak_center_binned_y'),
            active_computation_config=None, force_recompute=False, debug_print=debug_print)
        
        # Filter the summits, compute peak-counts, etc:
        active_eloy_analysis = None
        filtered_summits_analysis_df, pf_peak_counts_map = cls._build_filtered_summits_analysis_results(
            xbin, ybin, np.arange(1, len(xbin)), np.arange(1, len(ybin)),
            posterior_peaks_df, active_eloy_analysis,
            slice_level_multiplier=0.5,
            minimum_included_peak_height=minimum_included_peak_height,
            debug_print=debug_print)
        
        pf_peak_counts_map_blurred = uniform_filter(pf_peak_counts_map.astype('float'), size=uniform_blur_size, mode='constant')
        pf_peak_counts_map_blurred_gaussian = gaussian_filter(pf_peak_counts_map.astype('float'), sigma=gaussian_blur_sigma)
        pf_peak_counts_results = DynamicParameters(raw=pf_peak_counts_map,
                                                uniform_blurred=pf_peak_counts_map_blurred,
                                                gaussian_blurred=pf_peak_counts_map_blurred_gaussian)
        
        return DynamicParameters(xx=xbin_centers, yy=ybin_centers, results=out_results,
                                flat_peaks_df=posterior_peaks_df,
                                filtered_flat_peaks_df=filtered_summits_analysis_df,
                                peak_counts=pf_peak_counts_results)




# ==================================================================================================================================================================================================================================================================================== #
# Metrics/Scoring                                                                                                                                                                                                                                                                      #
# ==================================================================================================================================================================================================================================================================================== #

class PeakPromenenceMetrics:
    """

    Usage:
        from pyphoplacecellanalysis.External.peak_prominence2d import PeakPromenenceMetrics


        # After calling _compute_single_posterior_slab:
        epoch_idx, t_idx, posterior_peaks_df, slab_result_dict = _compute_single_posterior_slab(...)

        # Score the slab
        score_result = score_slab_quality(
            slab_result_dict=slab_result_dict,
            posterior_peaks_df=posterior_peaks_df,
            xbin_centers=xbin_centers,
            ybin_centers=ybin_centers,
            max_reasonable_peak_distance=50.0,  # cm, adjust for your maze
            close_peak_distance_threshold=5.0    # cm, adjust for bin spacing
        )

        if score_result['is_well_localized']:
            # This slab represents a specific decoded position
            print(f"Good slab at epoch {epoch_idx}, t {t_idx}: score = {score_result['overall_score']:.3f}")
        else:
            # Too diffuse or disparate peaks
            print(f"Reject slab at epoch {epoch_idx}, t {t_idx}: score = {score_result['overall_score']:.3f}")
            print(f"  Components: {score_result['score_components']}")


    """
    @classmethod
    def score_slab_quality(cls, slab_result_dict: dict, posterior_peaks_df: pd.DataFrame, xbin_centers: NDArray, ybin_centers: NDArray, 
                        max_reasonable_peak_distance: float = None, min_contour_size_threshold: float = 0.5,
                        close_peak_distance_threshold: float = None) -> dict:
        """
        Score the quality of a posterior slab based on peak prominence and spatial distribution.
        
        Returns a score dict with:
        - 'overall_score': float in [0, 1], higher = better (more localized, single position)
        - 'is_well_localized': bool, True if represents a specific decoded position
        - 'score_components': dict with individual component scores
        
        Parameters:
        -----------
        slab_result_dict : dict
            Contains 'peaks', 'slab', 'id_map', 'prominence_map', 'parent_map'
        posterior_peaks_df : pd.DataFrame
            DataFrame with peak information (one row per peak-slice combination)
        xbin_centers, ybin_centers : NDArray
            Spatial coordinates
        max_reasonable_peak_distance : float, optional
            Maximum reasonable distance between peaks (e.g., maze diagonal). 
            If None, uses 2 * max(xbin_centers.ptp(), ybin_centers.ptp())
        min_contour_size_threshold : float
            Multiplier for contour size threshold (relative to bin spacing)
        close_peak_distance_threshold : float, optional
            Distance below which peaks are considered "close" (same blob).
            If None, uses 3 * mean bin spacing
        """
        import numpy as np
        from scipy.spatial.distance import cdist
        
        peaks_dict = slab_result_dict['peaks']
        slab = slab_result_dict['slab']
        
        n_peaks = len(peaks_dict)
        if n_peaks == 0:
            return {
                'overall_score': 0.0,
                'is_well_localized': False,
                'score_components': {
                    'n_peaks': 0,
                    'dominant_prominence_score': 0.0,
                    'peak_clustering_score': 0.0,
                    'spatial_dispersion_score': 0.0,
                    'concentration_score': 0.0,
                    'diffuseness_penalty': 1.0
                }
            }
        
        # Compute spatial scales
        x_spacing = np.mean(np.diff(np.sort(xbin_centers))) if len(xbin_centers) > 1 else 1.0
        y_spacing = np.mean(np.diff(np.sort(ybin_centers))) if len(ybin_centers) > 1 else 1.0
        mean_bin_spacing = np.mean([x_spacing, y_spacing])
        
        if max_reasonable_peak_distance is None:
            max_reasonable_peak_distance = 2.0 * max(xbin_centers.ptp(), ybin_centers.ptp())
        
        if close_peak_distance_threshold is None:
            close_peak_distance_threshold = 3.0 * mean_bin_spacing
        
        # Extract peak information
        peak_centers = np.array([peak['center'] for peak in peaks_dict.values()])
        peak_heights = np.array([peak['height'] for peak in peaks_dict.values()])
        peak_prominences = np.array([peak['prominence'] for peak in peaks_dict.values()])
        peak_parents = np.array([peak['parent'] for peak in peaks_dict.values()])
        peak_ids = np.array(list(peaks_dict.keys()))
        
        # Sort by prominence (descending)
        sort_idx = np.argsort(peak_prominences)[::-1]
        peak_prominences_sorted = peak_prominences[sort_idx]
        peak_heights_sorted = peak_heights[sort_idx]
        peak_centers_sorted = peak_centers[sort_idx]
        
        # === COMPONENT 1: Dominant Peak Prominence Score ===
        # High prominence = distinct, well-separated peak
        max_prominence = peak_prominences_sorted[0]
        max_height = peak_heights_sorted[0]
        
        # Normalize prominence relative to peak height (prominence/height ratio)
        # Higher ratio = more distinct peak
        prominence_ratio = max_prominence / max_height if max_height > 0 else 0.0
        dominant_prominence_score = np.clip(prominence_ratio, 0.0, 1.0)
        
        # === COMPONENT 2: Peak Clustering Score ===
        # Check if peaks are close together (acceptable) vs far apart (bad)
        if n_peaks == 1:
            peak_clustering_score = 1.0  # Perfect: single peak
        else:
            # Compute pairwise distances
            pairwise_distances = cdist(peak_centers, peak_centers)
            # Remove diagonal (self-distances)
            pairwise_distances = pairwise_distances[np.triu_indices(n_peaks, k=1)]
            
            # Check parent relationships: peaks with same parent are likely part of same structure
            # Count how many peaks share the dominant peak as parent
            dominant_peak_id = peak_ids[sort_idx[0]]
            children_of_dominant = np.sum(peak_parents == dominant_peak_id)
            
            # Score based on:
            # 1. How many peaks are "close" (within threshold)
            # 2. How many share parent relationship with dominant peak
            close_peaks_ratio = np.sum(pairwise_distances < close_peak_distance_threshold) / len(pairwise_distances)
            parent_relationship_score = (children_of_dominant + 1) / n_peaks  # +1 for dominant itself
            
            # Combined: prefer many close peaks or peaks with parent relationships
            peak_clustering_score = 0.6 * close_peaks_ratio + 0.4 * parent_relationship_score
        
        # === COMPONENT 3: Spatial Dispersion Score ===
        # Penalize widely separated peaks (positional impossibility)
        if n_peaks == 1:
            spatial_dispersion_score = 1.0
        else:
            max_pairwise_distance = np.max(pairwise_distances)
            # Score: 1.0 if all peaks close, 0.0 if max distance > max_reasonable_peak_distance
            spatial_dispersion_score = 1.0 - np.clip(
                (max_pairwise_distance - close_peak_distance_threshold) / 
                (max_reasonable_peak_distance - close_peak_distance_threshold),
                0.0, 1.0
            )
        
        # === COMPONENT 4: Concentration Score ===
        # Check if probability mass is concentrated vs diffuse
        # Use contour sizes at a low threshold (e.g., 0.5 * peak height)
        # Get the lowest probe level from the DataFrame
        if len(posterior_peaks_df) > 0:
            # Get contours at lowest threshold for dominant peak
            dominant_peak_id = peak_ids[sort_idx[0]]
            dominant_peak_rows = posterior_peaks_df[
                (posterior_peaks_df['summit_idx'] == dominant_peak_id)
            ]
            
            if len(dominant_peak_rows) > 0:
                # Use the lowest slice level (highest multiplier = closer to peak)
                lowest_slice = dominant_peak_rows.loc[
                    dominant_peak_rows['slice_level_multiplier'].idxmin()
                ]
                
                # Get contour dimensions
                contour_width = lowest_slice['summit_slice_x_width']
                contour_height = lowest_slice['summit_slice_y_width']
                contour_area = contour_width * contour_height
                
                # Compare to expected size for a well-localized peak
                # Expected size ~ (2-3 bins)^2 for a tight peak
                expected_area = (2.5 * mean_bin_spacing) ** 2
                area_ratio = expected_area / contour_area if contour_area > 0 else 0.0
                
                # Score: 1.0 if contour is small (concentrated), 0.0 if very large (diffuse)
                concentration_score = np.clip(area_ratio, 0.0, 1.0)
            else:
                concentration_score = 0.5  # Default if no contour data
        else:
            concentration_score = 0.5
        
        # === COMPONENT 5: Diffuseness Penalty ===
        # Additional check: compare total slab variance to peak concentration
        slab_max = np.nanmax(slab)
        slab_mean = np.nanmean(slab)
        slab_std = np.nanstd(slab)
        
        # If std is high relative to max, distribution is diffuse
        # If most values are similar to max, it's concentrated
        concentration_ratio = (slab_max - slab_mean) / (slab_std + 1e-10)
        # Higher ratio = more concentrated
        diffuseness_penalty = 1.0 - np.clip(1.0 / (1.0 + concentration_ratio), 0.0, 1.0)
        
        # === COMBINE SCORES ===
        # Weighted combination (adjust weights based on your priorities)
        overall_score = (
            0.30 * dominant_prominence_score +      # How distinct is the main peak?
            0.25 * peak_clustering_score +          # Are peaks close together?
            0.20 * spatial_dispersion_score +       # Are peaks too far apart?
            0.15 * concentration_score +            # Is probability mass concentrated?
            0.10 * diffuseness_penalty              # Penalty for diffuse distributions
        )
        
        # Threshold for "well localized"
        # Adjust threshold based on your needs (0.6-0.7 is reasonable)
        is_well_localized = overall_score >= 0.65
        
        return {
            'overall_score': overall_score,
            'is_well_localized': is_well_localized,
            'score_components': {
                'n_peaks': n_peaks,
                'dominant_prominence_score': dominant_prominence_score,
                'peak_clustering_score': peak_clustering_score,
                'spatial_dispersion_score': spatial_dispersion_score,
                'concentration_score': concentration_score,
                'diffuseness_penalty': diffuseness_penalty,
                'max_pairwise_distance': np.max(pairwise_distances) if n_peaks > 1 else 0.0,
                'dominant_prominence': max_prominence,
                'dominant_height': max_height
            }
        }





# ==================================================================================================================================================================================================================================================================================== #
# Plotting/Figures                                                                                                                                                                                                                                                                     #
# ==================================================================================================================================================================================================================================================================================== #

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def plot_Prominence(xx, yy, slab, peaks, idmap, promap, parentmap, n_contour_levels=None, debug_print=False):
#     """Compatibility wrapper. Use PeakPromenenceDisplay.plot_Prominence instead."""
#     return PeakPromenenceDisplay.plot_Prominence(xx, yy, slab, peaks, idmap, promap, parentmap, n_contour_levels=n_contour_levels, debug_print=debug_print)


class PeakPromenenceDisplay:
    """ helper plot functions 

    from pyphoplacecellanalysis.External.peak_prominence2d import PeakPromenence, PeakPromenenceDisplay

    # Usage example:
        from pyphoplacecellanalysis.External.peak_prominence2d import PeakPromenence, PeakPromenenceDisplay

        # Compute peaks
        posterior_peaks = PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation(
            p_x_given_n_list=decoded_result.p_x_given_n_list,
            xbin_centers=decoded_result.xbin_centers,
            ybin_centers=decoded_result.ybin_centers,
            step=0.02,
            peak_height_multiplier_probe_levels=(0.5, 0.9),
            minimum_included_peak_height=0.2
        )

        # Plot single time bin
        plotter = PeakPromenenceDisplay.plot_prominence_peaks_3d_pyvista(
            posterior_peaks_result=posterior_peaks,
            p_x_given_n_list=decoded_result.p_x_given_n_list,
            epoch_idx=0,
            time_bin_idx=0,
            show_col_contours=True,
            show_probe_level_contours=True,
            probe_level_to_show=0.5,  # Only show 0.5x height contours
            opacity=0.7
        )

        plotter.show()

        # Or plot multiple time bins in a grid
        plotter_grid = PeakPromenenceDisplay.plot_prominence_peaks_3d_pyvista_grid(
            posterior_peaks_result=posterior_peaks,
            p_x_given_n_list=decoded_result.p_x_given_n_list,
            epoch_idx=0,
            time_bin_indices=[0, 5, 10, 15],
            n_cols=2
        )

        plotter_grid.show()
    """

    @classmethod
    def plot_Prominence(cls, xx, yy, slab, peaks, idmap, promap, parentmap, n_contour_levels=None, debug_print=False):
        """ simple test plot of the results calculated from getProminence.
        
        Inputs:
            n_contour_levels: should be an integer indicating the number of levels to display in the contour plot
            
        Usage:
        
            from pyphoplacecellanalysis.External.peak_prominence2d import getProminence, PeakPromenenceDisplay
            
            step = 0.2
            xx = active_pf_2D_dt.xbin_labels
            yy = active_pf_2D_dt.ybin_labels
            slab = active_pf_2D.ratemap.tuning_curves[3].T
            zmax = slab.max()
            peaks, idmap, promap, parentmap = getProminence(slab, step, ybin_centers=yy, xbin_centers=xx, min_area=None, include_edge=True, verbose=False)
            figure, (ax1, ax2, ax3, ax4) = PeakPromenenceDisplay.plot_Prominence(xx, yy, slab, peaks, idmap, promap, parentmap, debug_print=False)
        
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

    @classmethod
    def path_to_pyvista_polyline(cls, path, z_level):
        """Convert matplotlib Path to PyVista polyline at given z-level.
        Helper function to convert matplotlib Path to PyVista PolyData
        """
        import pyvista as pv

        vertices_2d = path.vertices
        # Close the path if not already closed
        if not np.allclose(vertices_2d[0], vertices_2d[-1]):
            vertices_2d = np.vstack([vertices_2d, vertices_2d[0:1]])
        
        # Add z-coordinate
        vertices_3d = np.column_stack([vertices_2d, np.full(len(vertices_2d), z_level)])
        
        # Create polyline
        polyline = pv.PolyData(vertices_3d)
        polyline.lines = np.array([len(vertices_3d)] + list(range(len(vertices_3d))))
        
        return polyline

    @classmethod
    def path_to_pyvista_mesh(cls, path, z_level, height_offset=0.01):
        """Convert matplotlib Path to PyVista mesh at given z-level with slight height.
        Helper function to create a filled contour mesh (extruded to z-level)
        """
        import pyvista as pv
        
        vertices_2d = path.vertices
        # Close the path if not already closed
        if not np.allclose(vertices_2d[0], vertices_2d[-1]):
            vertices_2d = np.vstack([vertices_2d, vertices_2d[0:1]])
        
        # Create polygon
        polygon = pv.PolyData(vertices_2d)
        polygon = polygon.delaunay_2d()
        
        # Extrude to create a thin mesh at z-level
        extruded = polygon.extrude((0, 0, height_offset))
        extruded.translate((0, 0, z_level), inplace=True)
        
        return extruded

    @classmethod
    def _plot_single_time_bin_pyvista(cls, plotter, posterior_peaks_result, p_x_given_n_list, epoch_idx, time_bin_idx, xx, yy, show_col_contours=True, show_probe_level_contours=True, probe_level_to_show=None, opacity=0.7, cmap='viridis', z_axis_scale: float=100.0, show_scalar_bar=True):
        """
        Core plotting logic for a single time bin using PyVista.
        
        Parameters:
        -----------
        plotter : pv.Plotter
            PyVista plotter instance
        posterior_peaks_result : DynamicParameters
            Result from `_perform_find_posterior_peaks_peak_prominence2d_computation`
        p_x_given_n_list : List[NDArray]
            Original posterior list used to generate peaks
        epoch_idx : int
            Which epoch to visualize
        time_bin_idx : int
            Which time bin to visualize
        xx : NDArray
            X bin centers
        yy : NDArray
            Y bin centers
        show_col_contours : bool
            Whether to show the col (key col) contours (default: True)
        show_probe_level_contours : bool
            Whether to show probe level contours (default: True)
        probe_level_to_show : float or None
            If specified, only show contours at this probe level multiplier (e.g., 0.5, 0.9)
            If None, show all probe levels (default: None)
        opacity : float
            Opacity of the posterior surface (default: 0.7)
        cmap : str
            Colormap for the posterior surface (default: 'viridis')
        z_axis_scale : float
            Scale factor for z-axis (default: 100.0)
        show_scalar_bar : bool
            Whether to show the scalar bar (default: True)
        
        Returns:
        --------
        actors : List
            List of mesh actors created
        """
        import pyvista as pv
        
        # Get the results for this epoch/time_bin
        result_key = (epoch_idx, time_bin_idx)
        if result_key not in posterior_peaks_result.results:
            return []  # Return empty list if no results for this time bin
        
        result = posterior_peaks_result.results[result_key]
        peaks_dict = result['peaks']
        slab = result['slab']  # This is already transposed (y, x)
        
        # Get the original posterior
        p_x_given_n = np.asarray(p_x_given_n_list[epoch_idx])
        if p_x_given_n.ndim == 3:
            posterior_2d = p_x_given_n[:, :, time_bin_idx]
        else:
            posterior_2d = p_x_given_n
        
        # Create meshgrid for surface
        # Use posterior_2d.T to match the transposed slab convention and ensure coordinate alignment
        XX, YY = np.meshgrid(xx, yy)
        ZZ = posterior_2d.T  # Transpose to match slab format (y, x)
        ZZ = (ZZ * z_axis_scale)
        
        actors = []
        
        # Create and add the posterior surface
        grid = pv.StructuredGrid(XX, YY, ZZ)
        grid_actor = plotter.add_mesh(grid, scalars=ZZ.flatten(), cmap=cmap, opacity=opacity, show_scalar_bar=show_scalar_bar, scalar_bar_args={'title': 'Posterior Probability'})
        actors.append(grid_actor)
        
        # Plot col contours (key col - the lowest contour for each peak)
        if show_col_contours:
            for peak_id, peak_info in peaks_dict.items():
                col_contour = peak_info.get('contour', None)
                if col_contour is not None:
                    col_level = peak_info.get('col_level', 0.0)
                    col_level = col_level * z_axis_scale
                    # Create polyline at col level
                    col_polyline = cls.path_to_pyvista_polyline(col_contour, col_level)
                    col_actor = plotter.add_mesh(col_polyline, color='red', line_width=3, label=f'Peak {peak_id} Col (prom={peak_info["prominence"]:.3f})')
                    actors.append(col_actor)
        
        # Plot probe level contours (slices at different height multipliers)
        if show_probe_level_contours:
            colors = ['cyan', 'yellow', 'magenta', 'green', 'orange']
            for peak_id, peak_info in peaks_dict.items():
                level_slices = peak_info.get('level_slices', {})
                peak_height = peak_info.get('height', 0.0)
                
                for slice_idx, (probe_level, slice_info) in enumerate(level_slices.items()):
                    # Filter by probe_level_to_show if specified
                    if probe_level_to_show is not None:
                        # Check if this probe level matches the desired multiplier
                        probe_multiplier = probe_level / peak_height
                        if not np.isclose(probe_multiplier, probe_level_to_show, atol=0.01):
                            continue
                    
                    contour = slice_info.get('contour', None)
                    if contour is not None:
                        color = colors[slice_idx % len(colors)]
                        # Create polyline at probe level (scale to match surface z-scaling)
                        probe_polyline = cls.path_to_pyvista_polyline(contour, probe_level * z_axis_scale)
                        probe_actor = plotter.add_mesh(probe_polyline, color=color, line_width=2,
                                                      label=f'Peak {peak_id} @ {probe_level/peak_height:.1f}x height')
                        actors.append(probe_actor)
        
        # Add peak centers as spheres
        for peak_id, peak_info in peaks_dict.items():
            center = peak_info.get('center', None)
            height = peak_info.get('height', 0.0)
            if center is not None:
                # Scale height to match surface z-scaling
                peak_sphere = pv.Sphere(radius=0.1, center=(center[0], center[1], height * z_axis_scale))
                sphere_actor = plotter.add_mesh(peak_sphere, color='white', label=f'Peak {peak_id} center')
                actors.append(sphere_actor)
        
        return actors

    @classmethod
    def plot_prominence_peaks_3d_pyvista(cls, posterior_peaks_result, p_x_given_n_list, epoch_idx=0, time_bin_idx=0, show_col_contours=True, show_probe_level_contours=True, probe_level_to_show=None, opacity=0.7, cmap='viridis', z_axis_scale: float=100.0):
        """
        Plot prominence peaks as 3D contours overlaying posteriors using PyVista.
        
        Parameters:
        -----------
        posterior_peaks_result : DynamicParameters
            Result from `_perform_find_posterior_peaks_peak_prominence2d_computation`
        p_x_given_n_list : List[NDArray]
            Original posterior list used to generate peaks
        epoch_idx : int
            Which epoch to visualize (default: 0)
        time_bin_idx : int
            Which time bin to visualize (default: 0)
        show_col_contours : bool
            Whether to show the col (key col) contours (default: True)
        show_probe_level_contours : bool
            Whether to show probe level contours (default: True)
        probe_level_to_show : float or None
            If specified, only show contours at this probe level multiplier (e.g., 0.5, 0.9)
            If None, show all probe levels (default: None)
        opacity : float
            Opacity of the posterior surface (default: 0.7)
        cmap : str
            Colormap for the posterior surface (default: 'viridis')
        
        Returns:
        --------
        plotter : pv.Plotter
            PyVista plotter object
        """
        import pyvista as pv
        import pyvistaqt as pvqt

        # Extract all available time_bin_idx values for the given epoch_idx
        available_time_bins = sorted([k[1] for k in posterior_peaks_result.results.keys() if k[0] == epoch_idx])
        if len(available_time_bins) == 0:
            raise ValueError(f"No results found for epoch_idx={epoch_idx}")
        
        # Initialize with first available if provided one doesn't exist
        if time_bin_idx not in available_time_bins:
            time_bin_idx = available_time_bins[0]
        
        # Get bin centers (these don't change with time_bin_idx)
        xx = posterior_peaks_result.xx
        yy = posterior_peaks_result.yy
        
        # Initialize plotter
        plotter = pvqt.BackgroundPlotter()
        
        # Store mesh actors for cleanup when updating
        mesh_actors = []
        first_update = True  # Track if this is the first update (for scalar bar)
        
        # Function to update the plot for a given time_bin_idx
        def _update_plot_for_time_bin(t_idx):
            """Update the plot to show data for the specified time_bin_idx."""
            nonlocal first_update
            
            # Clear existing meshes
            for actor in mesh_actors:
                plotter.remove_actor(actor)
            mesh_actors.clear()
            
            # Use the factored-out plotting method
            new_actors = cls._plot_single_time_bin_pyvista(
                plotter, posterior_peaks_result, p_x_given_n_list, epoch_idx, t_idx, xx, yy,
                show_col_contours=show_col_contours, show_probe_level_contours=show_probe_level_contours,
                probe_level_to_show=probe_level_to_show, opacity=opacity, cmap=cmap, z_axis_scale=z_axis_scale,
                show_scalar_bar=first_update
            )
            mesh_actors.extend(new_actors)
            first_update = False
        
        # Initial plot
        _update_plot_for_time_bin(time_bin_idx)
        
        # Set up the plotter (only once, not in update function)
        plotter.add_axes()
        plotter.add_legend()
        plotter.show_grid()
        plotter.set_background('black')
        
        # Add slider if multiple time bins available
        if len(available_time_bins) > 1:
            def slider_callback(value):
                """Callback function for slider widget."""
                idx = int(round(value))
                if 0 <= idx < len(available_time_bins):
                    t_idx = available_time_bins[idx]
                    _update_plot_for_time_bin(t_idx)
            
            # Get initial slider value
            initial_slider_value = available_time_bins.index(time_bin_idx)
            
            plotter.add_slider_widget(
                slider_callback,
                value=initial_slider_value,
                rng=[0, len(available_time_bins) - 1],
                title='Time Bin',
                pointa=(0.02, 0.02),
                pointb=(0.98, 0.02)
            )
        
        return plotter, _update_plot_for_time_bin


    # Alternative function that shows multiple time bins in a grid
    @function_attributes(short_name=None, tags=['NOT-FINISHEd', 'NOT-WORKING', 'NOT-TESTED'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-05 15:59', related_items=[])
    @classmethod
    def plot_prominence_peaks_3d_pyvista_grid(cls, posterior_peaks_result, p_x_given_n_list, epoch_idx=0, time_bin_indices=None, n_cols=3, show_col_contours=True, show_probe_level_contours=True, probe_level_to_show=None, **kwargs):
        """
        Plot multiple time bins in a grid layout.
        
        Parameters:
        -----------
        posterior_peaks_result : DynamicParameters
            Result from `_perform_find_posterior_peaks_peak_prominence2d_computation`
        p_x_given_n_list : List[NDArray]
            Original posterior list
        epoch_idx : int
            Which epoch to visualize
        time_bin_indices : List[int] or None
            Which time bins to show. If None, show all available.
        n_cols : int
            Number of columns in the grid
        show_col_contours : bool
            Whether to show the col (key col) contours (default: True)
        show_probe_level_contours : bool
            Whether to show probe level contours (default: True)
        probe_level_to_show : float or None
            If specified, only show contours at this probe level multiplier (e.g., 0.5, 0.9)
            If None, show all probe levels (default: None)
        **kwargs : dict
            Additional arguments (cmap, opacity, etc.)
        
        Returns:
        --------
        plotter : pv.Plotter
            PyVista plotter with subplots
        """
        import pyvista as pv
        import pyvistaqt as pvqt

        # Helper function to convert matplotlib Path to PyVista PolyData

        
        # Determine which time bins to show
        if time_bin_indices is None:
            # Get all available time bins for this epoch
            available_keys = [k for k in posterior_peaks_result.results.keys() if k[0] == epoch_idx]
            time_bin_indices = sorted(list(set([k[1] for k in available_keys])))
        
        n_plots = len(time_bin_indices)
        n_rows = int(np.ceil(n_plots / n_cols))
        
        # plotter = pv.Plotter(shape=(n_rows, n_cols))
        plotter = pvqt.MultiPlotter(nrows=n_rows, ncols=n_cols)

        # Get default kwargs
        cmap = kwargs.get('cmap', 'viridis')
        opacity = kwargs.get('opacity', 0.7)
        z_axis_scale = kwargs.get('z_axis_scale', 100.0)
        colors = kwargs.get('colors', ['cyan', 'yellow', 'magenta', 'green', 'orange'])
        
        for plot_idx, t_idx in enumerate(time_bin_indices):
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            plotter.subplot(row, col)
            # plotter[row, col]
            
            # Create individual plot
            result_key = (epoch_idx, t_idx)
            if result_key in posterior_peaks_result.results:
                result = posterior_peaks_result.results[result_key]
                peaks_dict = result['peaks']
                slab = result['slab']
                
                xx = posterior_peaks_result.xx
                yy = posterior_peaks_result.yy
                
                # Get posterior
                p_x_given_n = np.asarray(p_x_given_n_list[epoch_idx])
                if p_x_given_n.ndim == 3:
                    posterior_2d = p_x_given_n[:, :, t_idx]
                else:
                    posterior_2d = p_x_given_n
                
                # Create surface
                # Use posterior_2d.T to match the transposed slab convention and ensure coordinate alignment
                XX, YY = np.meshgrid(xx, yy)
                ZZ = posterior_2d.T  # Transpose to match slab format (y, x)
                ZZ = (ZZ * z_axis_scale)
                
                grid = pv.StructuredGrid(XX, YY, ZZ)
                plotter.add_mesh(grid, scalars=ZZ.flatten(), cmap=cmap, opacity=opacity)
                
                # Plot col contours (key col - the lowest contour for each peak)
                if show_col_contours:
                    for peak_id, peak_info in peaks_dict.items():
                        col_contour = peak_info.get('contour', None)
                        if col_contour is not None:
                            col_level = peak_info.get('col_level', 0.0)
                            # Scale col level to match surface z-scaling
                            col_level = col_level * z_axis_scale
                            # Create polyline at col level
                            col_polyline = cls.path_to_pyvista_polyline(col_contour, col_level)
                            plotter.add_mesh(col_polyline, color='red', line_width=3)
                
                # Plot probe level contours (slices at different height multipliers)
                if show_probe_level_contours:
                    for peak_id, peak_info in peaks_dict.items():
                        level_slices = peak_info.get('level_slices', {})
                        peak_height = peak_info.get('height', 0.0)
                        
                        for slice_idx, (probe_level, slice_info) in enumerate(level_slices.items()):
                            # Filter by probe_level_to_show if specified
                            if probe_level_to_show is not None:
                                # Check if this probe level matches the desired multiplier
                                probe_multiplier = probe_level / peak_height
                                if not np.isclose(probe_multiplier, probe_level_to_show, atol=0.01):
                                    continue
                            
                            contour = slice_info.get('contour', None)
                            if contour is not None:
                                color = colors[slice_idx % len(colors)]
                                # Create polyline at probe level (scale to match surface z-scaling)
                                probe_polyline = cls.path_to_pyvista_polyline(contour, probe_level * z_axis_scale)
                                plotter.add_mesh(probe_polyline, color=color, line_width=2)
                
                # Add peak centers as spheres
                for peak_id, peak_info in peaks_dict.items():
                    center = peak_info.get('center', None)
                    height = peak_info.get('height', 0.0)
                    if center is not None:
                        # Scale height to match surface z-scaling
                        peak_sphere = pv.Sphere(radius=0.1, center=(center[0], center[1], height * z_axis_scale))
                        plotter.add_mesh(peak_sphere, color='white')
                
                plotter.add_text(f'Time bin {t_idx}', font_size=10)
                plotter.add_axes()
        
        plotter.link_views()  # Link camera views
        return plotter








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
    figure, (ax1, ax2, ax3, ax4) = PeakPromenenceDisplay.plot_Prominence(xx, yy, slab, peaks, idmap, promap, parentmap, debug_print=False)
    
    from pprint import pprint
    pprint(peaks)

    figure.show()
    plt.show()
