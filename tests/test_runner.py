import unittest
from runner import Step, Runner


class FakeStep(Step):
    def __init__(self, name='', output=None):
        super().__init__(name)
        self._input = None
        self.output = output

    @property
    def input(self):
        return self._input

    def run(self, input=None):
        self._executed = True
        self._input = input
        return self.output


class TestRunner(unittest.TestCase):
    def test_run(self):
        output_step_1 = 'output_step_1'
        output_step_2 = 'output_step_2'
        step1 = FakeStep(output=output_step_1)
        step2 = FakeStep(output=output_step_2)
        runner = Runner([step1, step2])
        runner.run()
        self.assertTrue(step1.executed)
        self.assertTrue(step2.executed)
        self.assertEqual(step2.input, output_step_1)

    def test_json_factory(self):
        json = {
            'name':
            'runner',
            'steps': [{
                'module': '__main__',
                'class': 'FakeStep',
                'name': 'step_0'
            }, {
                'module': '__main__',
                'class': 'FakeStep',
                'name': 'step_1'
            }]
        }
        runner = Runner.from_json(json)
        step_0, step_1 = runner.steps
        self.assertTrue(isinstance(step_0, FakeStep))
        self.assertTrue(isinstance(step_1, FakeStep))
        self.assertEqual(step_0.name, 'step_0')
        self.assertEqual(step_1.name, 'step_1')


if __name__ == '__main__':
    unittest.main()
