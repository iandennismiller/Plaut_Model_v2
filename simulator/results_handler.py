from simulator.results import Results
from common.constants import WordTypes


class ResultsHandler:
    def __init__(self, config):
        self.config = config

        self.training_loss = None
        self.plaut_accuracy = None
        self.anchor_accuracy = None
        self.probe_accuracy = None
        self.running_time = None
        self.output_data = None
        self.hl_activation_data = None
        self.ol_activation_data = None
        self.model_weights = None

        if self.config.Outputs.plotting['loss']:
            self.training_loss = Results(results_dir=self.config.General.rootdir + "/Training Loss",
                                         config=self.config,
                                         title="Training Loss",
                                         labels=("Epoch", "Loss"))
        if self.config.Outputs.plotting['plaut_acc']:
            self.plaut_accuracy = Results(results_dir=self.config.General.rootdir + "/Training Accuracy",
                                          config=self.config,
                                          title="Training Accuracy",
                                          labels=("Epoch", "Accuracy"),
                                          columns=WordTypes.plaut_types)
        if self.config.Outputs.plotting['anchor_acc']:
            self.anchor_accuracy = Results(results_dir=self.config.General.rootdir + "/Anchor Accuracy",
                                           config=self.config,
                                           title="Anchor Accuracy",
                                           labels=("Epoch", "Accuracy"),
                                           columns=WordTypes.anchor_types)
        if self.config.Outputs.plotting['probe_acc']:
            self.probe_accuracy = Results(results_dir=self.config.General.rootdir + "/Probe Accuracy",
                                          config=self.config,
                                          title="Probe Accuracy",
                                          labels=("Epoch", "Accuracy"),
                                          columns=WordTypes.probe_types)
        if self.config.Outputs.plotting['running_time']:
            self.running_time = Results(results_dir=self.config.General.rootdir,
                                     config=self.config,
                                     title="Running Time",
                                     labels=("Epoch", "Time (s)"))
        if self.config.Outputs.sim_results:
            self.output_data = Results(results_dir=self.config.General.rootdir,
                                       config=self.config,
                                       title="Simulation Results",
                                       labels=('Epoch', ""),
                                       columns=['example_id', 'orth', 'phon', 'category', 'correct', 'anchors_added'])
        if self.config.Outputs.hidden_activations:
            self.hl_activation_data = Results(results_dir=self.config.General.rootdir,
                                              config=self.config,
                                              title="Hidden Layer Activations",
                                              columns=['orth', 'category', 'activation'])
        if self.config.Outputs.output_activations:
            self.ol_activation_data = Results(results_dir=self.config.General.rootdir,
                                              config=self.config,
                                              title="Output Layer Activations",
                                              columns=['orth', 'category', 'activation'])
        if self.config.Outputs.weights:
            self.model_weights = Results(results_dir=self.config.General.rootdir,
                                         config=self.config,
                                         title="Model Weights",
                                         columns=['weights'])

    def add_data(self, category, epoch, data):
        if category == 'plaut_accuracy' and self.plaut_accuracy is not None:
            self.plaut_accuracy.append_row(epoch, data)

        if category == 'anchor_accuracy' and self.anchor_accuracy is not None:
            self.anchor_accuracy.append_row(epoch, data)

        if category == 'probe_accuracy' and self.probe_accuracy is not None:
            self.probe_accuracy.append_row(epoch, data)

        if category == 'loss' and self.training_loss is not None:
            self.training_loss.append_row(epoch, data)

        if category == 'running_time' and self.running_time is not None:
            self.running_time.append_row(epoch, data)

        if category == 'activations':
            word_type, word_data, (hl_activations, ol_activations) = data
            if self.hl_activation_data is not None and epoch % self.config.Outputs.hidden_activations[word_type] == 0:
                self.hl_activation_data.add_rows([epoch] * hl_activations.shape[0], {
                    'orth': word_data['orth'],
                    'category': word_data['type'],
                    'activation': hl_activations.tolist()
                })
            if self.ol_activation_data is not None and epoch % self.config.Outputs.output_activations[word_type] == 0:
                self.ol_activation_data.add_rows([epoch] * ol_activations.shape[0], {
                    'orth': word_data['orth'],
                    'category': word_data['type'],
                    'activation': ol_activations.tolist()
                })

        if category == 'weights' and self.model_weights is not None and epoch % self.config.Outputs.weights == 0:
            self.model_weights.append_row(epoch, data)


    def create_training_plots(self, epoch):
        if epoch % self.config.Outputs.plotting['loss'] == 0:
            self.training_loss.line_plot()
        if epoch % self.config.Outputs.plotting['plaut_acc'] == 0:
            self.plaut_accuracy.line_plot()
        if epoch % self.config.Outputs.plotting['anchor_acc'] == 0:
            self.anchor_accuracy.line_plot(mapping=WordTypes.anchor_mapping)
        if epoch % self.config.Outputs.plotting['probe_acc'] == 0:
            self.probe_accuracy.line_plot(mapping=WordTypes.probe_mapping)

    def create_final_plots(self):
        if self.plaut_accuracy is not None:
            self.plaut_accuracy.bar_plot()
        if self.anchor_accuracy is not None:
            self.anchor_accuracy.bar_plot()
        if self.probe_accuracy is not None:
            self.probe_accuracy.bar_plot()
        if self.running_time is not None:
            self.running_time.line_plot()

    def save_data(self):
        if self.hl_activation_data is not None:
            self.hl_activation_data.save_data(index_label='epoch')
        if self.ol_activation_data is not None:
            self.ol_activation_data.save_data(index_label='epoch')
        if self.model_weights is not None:
            self.model_weights.save_data(index_label='epoch', save_type='pickle')
