#  import unittest
#  from typing import Dict
#  from runner import Step, Runner, ClassifierStep
#  from classifiers import Classifier
#  from test_commons import DummySlide
#  from commons import Patch, PatchCollection, PandasPatchCollection
#
#
#  class FakeStep(Step):
#      def __init__(self, name='', output=None):
#          super().__init__(name)
#          self._input = None
#          self.output = output
#
#      def run(self, input=None):
#          self._executed = True
#          self._input = input
#          return self.output
#
#
#  class DummyClassifier(Classifier):
#      @staticmethod
#      def create(name):
#          return DummyClassifier(name)
#
#      def __init__(self, name):
#          self.name = name
#
#      def classify_patch(self, patch: Patch) -> Dict:
#          return {self.name: 1}
#
#
#  class TestRunner(unittest.TestCase):
#      def test_run(self):
#          _input = 'input'
#          output_step_1 = 'output_step_1'
#          output_step_2 = 'output_step_2'
#          step1 = FakeStep(output=output_step_1)
#          step2 = FakeStep(output=output_step_2)
#          runner = Runner([step1, step2], 'test', _input)
#          runner.run()
#          self.assertTrue(step1.executed)
#          self.assertTrue(step2.executed)
#          self.assertEqual(step1.input, _input)
#          self.assertEqual(step2.input, output_step_1)
#
#      def test_run_classifier_step(self):
#          classifier_0, classifier_1 = (DummyClassifier('c0'),
#                                        DummyClassifier('c1'))
#          classifier_step_0, classifier_step_1 = (ClassifierStep(classifier_0),
#                                                  ClassifierStep(classifier_1))
#
#          slide_size = (200, 100)
#          slide = DummySlide('slide', slide_size)
#          patch_size = (10, 10)
#          patch_collection = PandasPatchCollection(slide, patch_size)
#          runner = Runner([classifier_step_0, classifier_step_1], 'test',
#                          patch_collection)
#          output = runner.run()
#          self.assertTrue(classifier_0.name in output.features)
#          self.assertTrue(classifier_1.name in output.features)
#
#      def test_json_factory(self):
#          json = {
#              'name':
#              'runner',
#              'steps': [{
#                  'module': '__main__',
#                  'class': 'FakeStep',
#                  'name': 'step_0'
#              }, {
#                  'module': '__main__',
#                  'class': 'FakeStep',
#                  'name': 'step_1'
#              }]
#          }
#          runner = Runner.from_json(json)
#          step_0, step_1 = runner.steps
#          self.assertTrue(isinstance(step_0, FakeStep))
#          self.assertTrue(isinstance(step_1, FakeStep))
#          self.assertEqual(step_0.name, 'step_0')
#          self.assertEqual(step_1.name, 'step_1')
#
#
#  class TestClassifierStep(unittest.TestCase):
#      def test_step(self):
#          classifier = DummyClassifier('test')
#          classifier_step = ClassifierStep(classifier)
#
#          slide_size = (200, 100)
#          slide = DummySlide('slide', slide_size)
#          patch_size = (10, 10)
#          patch_collection = PandasPatchCollection(slide, patch_size)
#          output = classifier_step.run(patch_collection)
#          self.assertTrue(isinstance(output, PatchCollection))
#          self.assertTrue(classifier.name in output.features)
#
#
#  if __name__ == '__main__':
#      unittest.main()
