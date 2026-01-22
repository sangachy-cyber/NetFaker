import unittest

from cases_background_utilization import ExampleBackgroundUtilization
from cases_bandwidth import ExampleBandwidth
from cases_classifier import ExampleClassifier
from cases_corruption import ExampleCorruption
from cases_delay import ExampleDelay
from cases_duplication import ExampleDuplication
from cases_filter import ExamplePixelFilter
from cases_frameoverhead import ExampleFrameOverhead
from cases_loss import ExampleLoss
from cases_modify import ExampleModify
from cases_mtu import ExampleMTU
from cases_queue_limit import ExampleQueueLimit
from cases_reordering import ExampleReordering


def load_tests():
    suite = unittest.TestSuite()

    suite.addTest(unittest.makeSuite(ExampleBackgroundUtilization))
    suite.addTest(unittest.makeSuite(ExampleBandwidth))
    suite.addTest(unittest.makeSuite(ExampleClassifier))
    suite.addTest(unittest.makeSuite(ExampleCorruption))
    suite.addTest(unittest.makeSuite(ExampleDelay))
    suite.addTest(unittest.makeSuite(ExampleDuplication))
    suite.addTest(unittest.makeSuite(ExampleFrameOverhead))
    suite.addTest(unittest.makeSuite(ExampleLoss))
    suite.addTest(unittest.makeSuite(ExampleModify))
    suite.addTest(unittest.makeSuite(ExampleMTU))
    suite.addTest(unittest.makeSuite(ExampleQueueLimit))
    suite.addTest(unittest.makeSuite(ExampleReordering))
    suite.addTest(unittest.makeSuite(ExamplePixelFilter))

    return suite


if __name__ == "__main__":
    # Create a test loader and runner
    loader = unittest.TestLoader()
    runner = unittest.TextTestRunner(verbosity=2)

    # Load the test suite
    suite = load_tests()

    # Run the test suite
    runner.run(suite)
