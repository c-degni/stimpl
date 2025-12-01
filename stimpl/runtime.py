from typing import Any, Tuple, Optional

from stimpl.expression import *
from stimpl.types import *
from stimpl.errors import *

"""
Interpreter State
"""


class State(object):
    def __init__(self, variable_name: str, variable_value: Expr, variable_type: Type, next_state: 'State') -> None:
        self.variable_name = variable_name
        self.value = (variable_value, variable_type)
        self.next_state = next_state

    def copy(self) -> 'State':
        variable_value, variable_type = self.value
        return State(self.variable_name, variable_value, variable_type, self.next_state)

    def set_value(self, variable_name: str, variable_value: Expr, variable_type: Type):
        return State(variable_name, variable_value, variable_type, self)

    def get_value(self, variable_name: str) -> Any:
        # States are implemented as a linked list with the current state at the head. Therefore to get the
        # the value of a variable in our current state, we check the value of the variable we are searching
        # for in the current state and then previous states recursively.
        if self.variable_name == variable_name:
            # Return the value of variable when found which will be the most updated value of variable
            return self.value
        elif self.next_state is not None:
            # Continue to recursively search for the variable in previous states as long as a previous state exists
            return self.next_state.get_value(variable_name)
        
        # At this point the variable was not found in any state (not declared)
        return None

    def __repr__(self) -> str:
        return f"{self.variable_name}: {self.value}, " + repr(self.next_state)


class EmptyState(State):
    def __init__(self):
        pass

    def copy(self) -> 'EmptyState':
        return EmptyState()

    def get_value(self, variable_name: str) -> None:
        return None

    def __repr__(self) -> str:
        return ""


"""
Main evaluation logic!
"""
def evaluate(expression: Expr, state: State) -> Tuple[Optional[Any], Type, State]:
    match expression:
        case Ren():
            return (None, Unit(), state)

        case IntLiteral(literal=l):
            return (l, Integer(), state)

        case FloatingPointLiteral(literal=l):
            return (l, FloatingPoint(), state)

        case StringLiteral(literal=l):
            return (l, String(), state)

        case BooleanLiteral(literal=l):
            return (l, Boolean(), state)

        case Print(to_print=to_print):
            printable_value, printable_type, new_state = evaluate(
                to_print, state)

            match printable_type:
                case Unit():
                    print("Unit")
                case _:
                    print(f"{printable_value}")

            return (printable_value, printable_type, new_state)

        case Sequence(exprs=exprs) | Program(exprs=exprs):
            # For a sequence/program, we are simply evaluating a list of given expressions while updating the state
            # accordingly.
            if len(exprs) == 0:
                # If the sequence/program is empty (contains no expressions), the configuration we return has value and type
                # unit, along with the current state.
                return (None, Unit(), state)
            
            for expr in exprs:
                # Evaluate every expression in the sequence/program in order, while updating our state as we go. Also track 
                # the track the the value and type of each expression for record.
                result, result_type, state = evaluate(expr, state)

            # Use the record of type and value from the latest expression as the type and value of the overall sequence/program.
            return (result, result_type, state)

        case Variable(variable_name=variable_name):
            value = state.get_value(variable_name)
            if value == None:
                raise InterpSyntaxError(
                    f"Cannot read from {variable_name} before assignment.")
            # Now that we know the result from `get_value` is not None,
            # we can look at the (v, tau) pieces of `value` that we know
            # forms its return value.
            variable_value, variable_type = value
            return (variable_value, variable_type, state)

        case Assign(variable=variable, value=value):

            value_result, value_type, new_state = evaluate(value, state)

            variable_from_state = new_state.get_value(variable.variable_name)
            _, variable_type = variable_from_state if variable_from_state else (
                None, None)

            if value_type != variable_type and variable_type != None:
                raise InterpTypeError(f"""Mismatched types for Assignment:
            Cannot assign {value_type} to {variable_type}""")

            new_state = new_state.set_value(
                variable.variable_name, value_result, value_type)
            return (value_result, value_type, new_state)

        case Add(left=left, right=right):
            result = 0
            left_result, left_type, new_state = evaluate(left, state)
            right_result, right_type, new_state = evaluate(right, new_state)

            if left_type != right_type:
                raise InterpTypeError(f"""Mismatched types for Add:
            Cannot add {left_type} to {right_type}""")

            match left_type:
                case Integer() | String() | FloatingPoint():
                    result = left_result + right_result
                case _:
                    raise InterpTypeError(f"""Cannot add {left_type}s""")

            return (result, left_type, new_state)

        case Subtract(left=left, right=right):
            # Initialize our result variable and obtain information of our left and right operands by evaluating
            # them (since EVERYTHING is an expression).
            result = 0
            left_result, left_type, new_state = evaluate(left, state)
            right_result, right_type, new_state = evaluate(right, new_state)

            # Check to make sure the operands are the same type because operands to binary operators must have the
            # same type.
            if left_type != right_type:
                raise InterpTypeError(f"""Mismatched types for Subtract:
                    Cannot subtract {right_type} from {left_type}""")

            # Match operator behavior with its given operand type. If operand behavior is not defined for that type, throw an
            # error.
            match left_type:
                case Integer() | FloatingPoint():
                    result = left_result - right_result
                case _:
                    raise InterpTypeError(f"""Cannot subtract {left_type}s""")

            # Return the resulting configuration as well as the state.
            return (result, left_type, new_state)

        case Multiply(left=left, right=right):
            # Initialize our result variable and obtain information of our left and right operands by evaluating
            # them (since EVERYTHING is an expression).
            result = 0
            left_result, left_type, new_state = evaluate(left, state)
            right_result, right_type, new_state = evaluate(right, new_state)

            # Check to make sure the operands are the same type because operands to binary operators must have the
            # same type.
            if left_type != right_type:
                raise InterpTypeError(f"""Mismatched types for Multiply:
                    Cannot multiply {left_type} and {right_type}""")

            # Match operator behavior with its given operand type. If operand behavior is not defined for that type, throw an
            # error.
            match left_type:
                case Integer() | FloatingPoint():
                    result = left_result * right_result
                case _:
                    raise InterpTypeError(f"""Cannot multiply {left_type}s""")

            # Return the resulting configuration as well as the state.
            return (result, left_type, new_state)

        case Divide(left=left, right=right):
            # Initialize our result variable and obtain information of our left and right operands by evaluating
            # them (since EVERYTHING is an expression).
            result = 0
            left_result, left_type, new_state = evaluate(left, state)
            right_result, right_type, new_state = evaluate(right, new_state)

            # Check to make sure the operands are the same type because operands to binary operators must have the
            # same type.
            if left_type != right_type:
                raise InterpTypeError(f"""Mismatched types for Divide:
                    Cannot divide {left_type} from {right_type}""")
            
            # To address divide by zero errors, we check if the right operand (divisor) is "zero" respectively and
            # throw an error.
            if isinstance(right_type, Integer) and right_result == 0 or isinstance(right_type, FloatingPoint) and right_result == 0.0:
                raise InterpMathError(f"""Attempted to divide by {right_result}
                    which results in undefined behavior""")

            # Match operator behavior with its given operand type. If operand behavior is not defined for that type, throw an
            # error.
            match left_type:
                case Integer():
                    result = left_result // right_result
                case FloatingPoint():
                    result = left_result / right_result
                case _:
                    raise InterpTypeError(f"""Cannot divide {left_type}s""")
            
            # Return the resulting configuration as well as the state.
            return (result, left_type, new_state)

        case And(left=left, right=right):
            left_value, left_type, new_state = evaluate(left, state)
            right_value, right_type, new_state = evaluate(right, new_state)

            if left_type != right_type:
                raise InterpTypeError(f"""Mismatched types for And:
                    Cannot evaluate {left_type} and {right_type}""")
            
            match left_type:
                case Boolean():
                    result = left_value and right_value
                case _:
                    raise InterpTypeError(
                        "Cannot perform logical and on non-boolean operands.")

            return (result, left_type, new_state)

        case Or(left=left, right=right):
            # Obtain information of our left and right operands by evaluating them (since EVERYTHING is an expression).
            left_value, left_type, new_state = evaluate(left, state)
            right_value, right_type, new_state = evaluate(right, new_state)

            # Check to make sure the operands are the same type because operands to binary operators must have the
            # same type.
            if left_type != right_type:
                raise InterpTypeError(f"""Mismatched types for Or:
                    Cannot evaluate {left_type} and {right_type}""")
            
            # Match operator behavior with its given operand type. If operand behavior is not defined for that type, throw an
            # error. In this case, behavior is only defined for type bool.
            match left_type:
                case Boolean():
                    result = left_value or right_value
                case _:
                    raise InterpTypeError(
                        "Cannot perform logical or on non-boolean operands.")
                
            # Return the resulting configuration as well as the state.
            return (result, left_type, new_state)

        case Not(expr=expr):
            # Obtain information of our operand by evaluating it (since EVERYTHING is an expression).
            value, value_type, new_state = evaluate(expr, state)

            # Match operator behavior with its given operand type. If operand behavior is not defined for that type, throw an
            # error. In this case, behavior is only defined for type bool.
            match value_type:
                case Boolean():
                    result = not value
                case _:
                    raise InterpTypeError(
                        "Cannot perform logical not on non-boolean operand.")
                
            # Return the resulting configuration as well as the state.
            return (result, value_type, new_state)

        case If(condition=condition, true=true, false=false):
            # Obtain information of our condition by evaluating it (since EVERYTHING is an expression).
            condition_value, condition_type, new_state = evaluate(condition, state)

            # Match operator behavior with its given operand type. If operand behavior is not defined for that type, throw an
            # error. In this case, behavior is only defined for type bool.
            match condition_type:
                case Boolean():
                    # Here, like with regular if-statements, we will evaluate the "true" expression if our condition has a
                    # value of true and the "false" expression if otherwise
                    if condition_value:
                        result, result_type, new_state = evaluate(true, new_state)
                    else:
                        result, result_type, new_state = evaluate(false, new_state)
                case _:
                    raise InterpTypeError(f"""Condition type is {condition_type}
                        and not boolean""")

            # Return the resulting configuration as well as the state (will be based on which expression was evaluated).
            return (result, result_type, new_state)

        case Lt(left=left, right=right):
            left_value, left_type, new_state = evaluate(left, state)
            right_value, right_type, new_state = evaluate(right, new_state)

            result = None

            if left_type != right_type:
                raise InterpTypeError(f"""Mismatched types for Lt:
                    Cannot compare {left_type} and {right_type}""")

            match left_type:
                case Integer() | Boolean() | String() | FloatingPoint():
                    result = left_value < right_value
                case Unit():
                    result = False
                case _:
                    raise InterpTypeError(
                        f"Cannot perform < on {left_type} type.")

            return (result, Boolean(), new_state)

        case Lte(left=left, right=right):
            # Initialize our result variable and obtain information of our left and right operands by evaluating
            # them (since EVERYTHING is an expression).
            left_value, left_type, new_state = evaluate(left, state)
            right_value, right_type, new_state = evaluate(right, new_state)

            result = None

            # Check to make sure the operands are the same type because operands to binary operators must have the
            # same type.
            if left_type != right_type:
                raise InterpTypeError(f"""Mismatched types for Lte:
                    Cannot compare {left_type} and {right_type}""")

            # Match operator behavior with its given operand type. If operand behavior is not defined for that type, throw an
            # error.
            match left_type:
                case Integer() | Boolean() | String() | FloatingPoint() | Ren():
                    result = left_value <= right_value
                case Unit():
                    result = True
                case _:
                    raise InterpTypeError(
                        f"Cannot perform <= on {left_type} type.")

            # Return the resulting configuration as well as the state.
            return (result, Boolean(), new_state)

        case Gt(left=left, right=right):
            # Initialize our result variable and obtain information of our left and right operands by evaluating
            # them (since EVERYTHING is an expression).
            left_value, left_type, new_state = evaluate(left, state)
            right_value, right_type, new_state = evaluate(right, new_state)

            result = None

            # Check to make sure the operands are the same type because operands to binary operators must have the
            # same type.
            if left_type != right_type:
                raise InterpTypeError(f"""Mismatched types for Gt:
            Cannot compare {left_type} and {right_type}""")

            # Match operator behavior with its given operand type. If operand behavior is not defined for that type, throw an
            # error.
            match left_type:
                case Integer() | Boolean() | String() | FloatingPoint():
                    result = left_value > right_value
                case Unit():
                    result = False
                case _:
                    raise InterpTypeError(
                        f"Cannot perform > on {left_type} type.")

            # Return the resulting configuration as well as the state.
            return (result, Boolean(), new_state)

        case Gte(left=left, right=right):
            # Initialize our result variable and obtain information of our left and right operands by evaluating
            # them (since EVERYTHING is an expression).
            left_value, left_type, new_state = evaluate(left, state)
            right_value, right_type, new_state = evaluate(right, new_state)

            result = None

            # Check to make sure the operands are the same type because operands to binary operators must have the
            # same type.
            if left_type != right_type:
                raise InterpTypeError(f"""Mismatched types for Gte:
            Cannot compare {left_type} and {right_type}""")

            # Match operator behavior with its given operand type. If operand behavior is not defined for that type, throw an
            # error.
            match left_type:
                case Integer() | Boolean() | String() | FloatingPoint():
                    result = left_value >= right_value
                case Unit():
                    result = True
                case _:
                    raise InterpTypeError(
                        f"Cannot perform >= on {left_type} type.")

            # Return the resulting configuration as well as the state.
            return (result, Boolean(), new_state)

        case Eq(left=left, right=right):
            # Initialize our result variable and obtain information of our left and right operands by evaluating
            # them (since EVERYTHING is an expression).
            left_value, left_type, new_state = evaluate(left, state)
            right_value, right_type, new_state = evaluate(right, new_state)

            result = None

            # Check to make sure the operands are the same type because operands to binary operators must have the
            # same type.
            if left_type != right_type:
                raise InterpTypeError(f"""Mismatched types for Eq:
            Cannot compare {left_type} and {right_type}""")

            # Match operator behavior with its given operand type. If operand behavior is not defined for that type, throw an
            # error.
            match left_type:
                case Integer() | Boolean() | String() | FloatingPoint():
                    result = left_value == right_value
                case Unit():
                    result = True
                case _:
                    raise InterpTypeError(
                        f"Cannot perform == on {left_type} type.")

            # Return the resulting configuration as well as the state.
            return (result, Boolean(), new_state)

        case Ne(left=left, right=right):
            # Initialize our result variable and obtain information of our left and right operands by evaluating
            # them (since EVERYTHING is an expression).
            left_value, left_type, new_state = evaluate(left, state)
            right_value, right_type, new_state = evaluate(right, new_state)

            result = None

            # Check to make sure the operands are the same type because operands to binary operators must have the
            # same type.
            if left_type != right_type:
                raise InterpTypeError(f"""Mismatched types for Ne:
            Cannot compare {left_type} and {right_type}""")

            # Match operator behavior with its given operand type. If operand behavior is not defined for that type, throw an
            # error.
            match left_type:
                case Integer() | Boolean() | String() | FloatingPoint() | Ren():
                    result = left_value != right_value
                case Unit():
                    result = False
                case _:
                    raise InterpTypeError(
                        f"Cannot perform != on {left_type} type.")

            # Return the resulting configuration as well as the state.
            return (result, Boolean(), new_state)

        case While(condition=condition, body=body):
            # Obtain information of our condition by evaluating it (since EVERYTHING is an expression).
            condition_value, condition_type, new_state = evaluate(condition, state)

            # Match operator behavior with its given operand type. If operand behavior is not defined for that type, throw an
            # error. In this case, behavior is only defined for type bool.
            match condition_type:
                case Boolean():
                    # Here, like with regular while-statements, we will evaluate the given "body" expression if our 
                    # while our condition is true, updating the state as we go along. Also we will reevaluate the condition
                    # to use as our next condition variable.
                    while condition_value:
                        _, _, new_state = evaluate(body, new_state)
                        condition_value, _, new_state = evaluate(condition, new_state)
                case _:
                    raise InterpTypeError(f"""Condition type is {condition_type}
                        and not boolean""")

            # Return false for while value and type bool as well as the state.
            return (False, Boolean(), new_state)

        case _:
            raise InterpSyntaxError("Unhandled!")
    pass


def run_stimpl(program, debug=False):
    state = EmptyState()
    program_value, program_type, program_state = evaluate(program, state)

    if debug:
        print(f"program: {program}")
        print(f"final_value: ({program_value}, {program_type})")
        print(f"final_state: {program_state}")

    return program_value, program_type, program_state
