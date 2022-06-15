""" Simple helper functions that are used to add Save/Restore state buttons to a ParameterTree

Known to be used by:
    PhoPy3DPositionAnalysis2021.LibrariesExamples.PyQtPlot.PyQtGraph_PipelineFilterParameterTree_Testing.py

"""

def default_parameters_save_restore_state_button_children():
    """ Builds the default save/restore state buttons for a parameter item """
    return {
        'name': 'Save/Restore functionality', 'type': 'group', 'children': [
        {'name': 'Save State', 'type': 'action'},
        {
            'name': 'Restore State', 'type': 'action', 'children': [
            {'name': 'Add missing items', 'type': 'bool', 'value': True},
            {'name': 'Remove extra items', 'type': 'bool', 'value': True},
        ]},
    ]}


def add_save_restore_btn_functionality(p):
    """ Sets up the Save/Restore State Button Functionality """
    # Save/Restore State Button Functionality:
    def save():
        global state
        state = p.saveState()

    def restore():
        global state
        add = p['Save/Restore functionality', 'Restore State', 'Add missing items']
        rem = p['Save/Restore functionality', 'Restore State', 'Remove extra items']
        p.restoreState(state, addChildren=add, removeChildren=rem)

    # Looks like here it tries to find the child named 'Save/Restore functionality' > 'Save State' to bind the buttons
    p.param('Save/Restore functionality', 'Save State').sigActivated.connect(save)
    p.param('Save/Restore functionality', 'Restore State').sigActivated.connect(restore)