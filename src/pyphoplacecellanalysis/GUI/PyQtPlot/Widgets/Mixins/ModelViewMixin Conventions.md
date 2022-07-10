# Model/View/Mixin Class Notes:

# ModelViewMixin Conventions.md

Each class makes use of either its own datasource or reuses one of the controller that it's a mixin on.

```python

from qtpy import QtCore

class MixinClassName:
    """ custom mixin class """
    
    @QtCore.Slot()
    def MixinClassName_on_init(self):
        """ perform any parameters setting/checking during init """
        pass

    @QtCore.Slot()
    def MixinClassName_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        pass


    @QtCore.Slot()
    def MixinClassName_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        pass

    @QtCore.Slot()
    def MixinClassName_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        pass

    @QtCore.Slot(float, float)
    def MixinClassName_on_window_update(self, new_start=None, new_end=None):
        """ called to perform updates when the active window changes. Redraw, recompute data, etc. """
        pass

    ############### Rate-Limited SLots ###############:
    ##################################################
    ## For use with pg.SignalProxy
    # using signal proxy turns original arguments into a tuple
    @QtCore.Slot(object)
    def MixinClassName_on_window_update_rate_limited(self, evt):
        self.MixinClassName_on_window_update(*evt)
        
```
