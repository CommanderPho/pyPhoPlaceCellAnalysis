from qtpy import QtCore, QtGui, QtWidgets


def create_custom_context_menu(owner):
	# # create context menu
	# self.popMenu = QtGui.QMenu(self)
	# self.popMenu.addAction(QtGui.QAction('test0', self))
	# self.popMenu.addAction(QtGui.QAction('test1', self))
	# self.popMenu.addSeparator()
	# self.popMenu.addAction(QtGui.QAction('test2', self))

	## create context menu
	owner.popMenu = QtWidgets.QMenu(owner)
	owner.popMenu.addAction(QtGui.QAction('test0', owner))
	owner.popMenu.addAction(QtGui.QAction('test1', owner))
	owner.popMenu.addSeparator()
	owner.popMenu.addAction(QtGui.QAction('test2', owner))
 