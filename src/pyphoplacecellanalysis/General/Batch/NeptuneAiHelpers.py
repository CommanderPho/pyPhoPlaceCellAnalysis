"""Deprecated compatibility shim — Neptune.ai has been replaced by Weights & Biases.

Import from `pyphoplacecellanalysis.General.Batch.WandbHelpers` instead.
"""
import warnings

warnings.warn(
	"NeptuneAiHelpers is deprecated; use pyphoplacecellanalysis.General.Batch.WandbHelpers (CoreWeave W&B / wandb).",
	DeprecationWarning,
	stacklevel=2,
)

from pyphoplacecellanalysis.General.Batch.WandbHelpers import (  # noqa: F401
	SessionDescriptorString,
	RunID,
	WandbKeyPath as NeptuneKeyPath,
	WandbProjectName as NeptuneProjectName,
	wandb_stringify_unsupported as stringify_unsupported,
	set_wandb_environment_variables as set_environment_variables,
	KnownWandbProjects,
	capture_print_output,
	strip_ansi_escape_sequences,
	clean_quotes,
	flatten_context_nested_dict,
	build_vscode_log_browsing_workspace,
	WandbRunHelper as AutoValueConvertingNeptuneRun,
	WandbRunCollectedResults as NeptuneRunCollectedResults,
	WandbLogger as Neptuner,
	PhoDibaProjectsWandbLogger as PhoDibaProjectsNeptuner,
	wandb_output_figures as neptune_output_figures,
)


class KnownNeptuneProjects:
	"""Deprecated alias of KnownWandbProjects with legacy method names."""

	@staticmethod
	def get_PhoDibaBatchProcessing_neptune_kwargs():
		return KnownWandbProjects.get_PhoDibaBatchProcessing_wandb_kwargs()


	@staticmethod
	def get_PhoDibaLongShortUpdated_neptune_kwargs():
		return KnownWandbProjects.get_PhoDibaLongShortUpdated_wandb_kwargs()


	@classmethod
	def get_PhoDibaBatchProcessing_neptuner(cls):
		return KnownWandbProjects.get_PhoDibaBatchProcessing_wandb_logger()


	@classmethod
	def get_PhoDibaLongShortUpdated_neptuner(cls):
		return KnownWandbProjects.get_PhoDibaLongShortUpdated_wandb_logger()
