from stimpl.expression import *
from stimpl.runtime import *
from stimpl.test import run_stimpl_sanity_tests

if __name__=='__main__':
  # program = Print(Assign(Variable("i"), StringLiteral("Hello, World")))
  # run_stimpl(program)
  run_stimpl_sanity_tests()
  
