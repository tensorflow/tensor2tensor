# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Algorithmic data generators for symbolic math tasks.

See go/symbolic-math-dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import random

# Dependency imports

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import sympy


class ExprOp(object):
  """Represents an algebraic operation, such as '+', '-', etc."""

  def __init__(self, symbol, precedence, associative=False):
    """Constructor.

    Args:
      symbol: The character which represents this operation, such as '+' for
          addition.
      precedence: Operator precedence. This will determine where parentheses
          are used.
      associative: If true, the order of the operands does not matter.
    """
    self.symbol = symbol
    self.precedence = precedence
    self.associative = associative

  def __str__(self):
    return self.symbol

  def __eq__(self, other):
    return isinstance(other, ExprOp) and self.symbol == other.symbol


class ExprNode(object):
  """A node in an expression tree.

  ExprNode always holds an operator. Leaves are strings.
  """

  def __init__(self, left, right, op):
    self.left = left
    self.right = right
    self.op = op
    left_depth = left.depth if isinstance(left, ExprNode) else 0
    right_depth = right.depth if isinstance(right, ExprNode) else 0
    self.depth = max(left_depth, right_depth) + 1

  def __str__(self):
    left_str = str(self.left)
    right_str = str(self.right)
    left_use_parens = (isinstance(self.left, ExprNode) and
                       self.left.op.precedence < self.op.precedence)
    right_use_parens = (isinstance(self.right, ExprNode) and
                        self.right.op.precedence <= self.op.precedence and
                        not (self.op.associative and self.right.op == self.op))
    left_final = "(" + left_str + ")" if left_use_parens else left_str
    right_final = "(" + right_str + ")" if right_use_parens else right_str
    return left_final + str(self.op) + right_final

  def is_in(self, expr):
    """Returns True if `expr` is a subtree."""
    if expr == self:
      return True
    is_in_left = is_in_expr(self.left, expr)
    is_in_right = is_in_expr(self.right, expr)
    return is_in_left or is_in_right


def is_in_expr(expr, find):
  """Returns True if `find` is a subtree of `expr`."""
  return expr == find or (isinstance(expr, ExprNode) and expr.is_in(find))


def random_expr_with_required_var(depth, required_var, optional_list, ops):
  """Generate a random expression tree with a required variable.

  The required variable appears exactly once in the expression.

  Args:
    depth: At least one leaf will be this many levels down from the top.
    required_var: A char. This char is guaranteed to be placed exactly once at
        a leaf somewhere in the tree. This is the var to solve for.
    optional_list: A list of chars. These chars are randomly selected as leaf
        values. These are constant vars.
    ops: A list of ExprOp instances.

  Returns:
    An ExprNode instance which is the root of the generated expression tree.
  """
  if not depth:
    if required_var:
      return required_var
    return str(optional_list[random.randrange(len(optional_list))])

  max_depth_side = random.randrange(2)
  other_side_depth = random.randrange(depth)

  required_var_side = random.randrange(2)

  left = random_expr_with_required_var(
      depth - 1 if max_depth_side else other_side_depth, required_var
      if required_var_side else None, optional_list, ops)
  right = random_expr_with_required_var(
      depth - 1 if not max_depth_side else other_side_depth, required_var
      if not required_var_side else None, optional_list, ops)

  op = ops[random.randrange(len(ops))]
  return ExprNode(left, right, op)


def random_expr(depth, vlist, ops):
  """Generate a random expression tree.

  Args:
    depth: At least one leaf will be this many levels down from the top.
    vlist: A list of chars. These chars are randomly selected as leaf values.
    ops: A list of ExprOp instances.

  Returns:
    An ExprNode instance which is the root of the generated expression tree.
  """
  if not depth:
    return str(vlist[random.randrange(len(vlist))])

  max_depth_side = random.randrange(2)
  other_side_depth = random.randrange(depth)

  left = random_expr(depth - 1
                     if max_depth_side else other_side_depth, vlist, ops)
  right = random_expr(depth - 1
                      if not max_depth_side else other_side_depth, vlist, ops)

  op = ops[random.randrange(len(ops))]
  return ExprNode(left, right, op)


def algebra_inverse_solve(left, right, var, solve_ops):
  """Solves for the value of the given var in an expression.

  See go/symbolic-math-dataset.

  Args:
    left: The root of the ExprNode tree on the left side of the equals sign.
    right: The root of the ExprNode tree on the right side of the equals sign.
    var: A char. The variable to solve for.
    solve_ops: A dictionary with the following properties.
        * For each operator in the expression, there is a rule that determines
          how to cancel out a value either to the left or the right of that
          operator.
        * For each rule, there is an entry in the dictionary. The key is two
          chars- the op char, and either 'l' or 'r' meaning rule for canceling
          out the left or right sides. For example, '+l', '+r', '-l', '-r'.
        * The value of each entry is a function with the following signature:
          (left, right, to_tree) -> (new_from_tree, new_to_tree)
          left- Expression on left side of the op.
          right- Expression on the right side of the op.
          to_tree- The tree on the other side of the equal sign. The canceled
              out expression will be moved here.
          new_from_tree- The resuling from_tree after the algebraic
              manipulation.
          new_to_tree- The resulting to_tree after the algebraic manipulation.

  Returns:
    The root of an ExprNode tree which holds the value of `var` after solving.

  Raises:
    ValueError: If `var` does not appear exactly once in the equation (which
        includes the left and right sides).
  """
  is_in_left = is_in_expr(left, var)
  is_in_right = is_in_expr(right, var)
  if is_in_left == is_in_right:
    if is_in_left:
      raise ValueError("Solve-variable '%s' is on both sides of the equation. "
                       "Only equations where the solve variable-appears once "
                       "are supported by this solver. Left: '%s', right: '%s'" %
                       (var, str(left), str(right)))
    else:
      raise ValueError("Solve-variable '%s' is not present in the equation. It "
                       "must appear once. Left: '%s', right: '%s'" %
                       (var, str(left), str(right)))

  from_tree = left if is_in_left else right
  to_tree = left if not is_in_left else right
  while from_tree != var:
    is_in_left = is_in_expr(from_tree.left, var)
    is_in_right = is_in_expr(from_tree.right, var)
    from_tree, to_tree = (solve_ops[str(from_tree.op)
                                    + ("l" if is_in_left else "r")](
                                        from_tree.left, from_tree.right,
                                        to_tree))
  return to_tree


def format_sympy_expr(sympy_expr, functions=None):
  """Convert sympy expression into a string which can be encoded.

  Args:
    sympy_expr: Any sympy expression tree or string.
    functions: Defines special functions. A dict mapping human readable string
        names, like "log", "exp", "sin", "cos", etc., to single chars. Each
        function gets a unique token, like "L" for "log".

  Returns:
    A string representation of the expression suitable for encoding as a
        sequence input.
  """
  if functions is None:
    functions = {}
  str_expr = str(sympy_expr)
  result = str_expr.replace(" ", "")
  for fn_name, char in six.iteritems(functions):
    result = result.replace(fn_name, char)
  return result


def generate_algebra_inverse_sample(vlist, ops, solve_ops, min_depth,
                                    max_depth):
  """Randomly generate an algebra inverse dataset sample.

  Given an input equation and variable, produce the expression equal to the
  variable.

  See go/symbolic-math-dataset.

  Args:
    vlist: Variable list. List of chars that can be used in the expression.
    ops: List of ExprOp instances. The allowed operators for the expression.
    solve_ops: See `solve_ops` documentation in `algebra_inverse_solve`.
    min_depth: Expression trees will not have a smaller depth than this. 0 means
        there is just a variable. 1 means there is one operation.
    max_depth: Expression trees will not have a larger depth than this. To make
        all trees have the same depth, set this equal to `min_depth`.

  Returns:
    sample: String representation of the input. Will be of the form
        'solve_var:left_side=right_side'.
    target: String representation of the solution.
  """
  side = random.randrange(2)
  left_depth = random.randrange(min_depth if side else 0, max_depth + 1)
  right_depth = random.randrange(min_depth if not side else 0, max_depth + 1)

  var_index = random.randrange(len(vlist))
  var = vlist[var_index]
  consts = vlist[:var_index] + vlist[var_index + 1:]

  left = random_expr_with_required_var(left_depth, var
                                       if side else None, consts, ops)
  right = random_expr_with_required_var(right_depth, var
                                        if not side else None, consts, ops)

  left_str = str(left)
  right_str = str(right)
  target = str(algebra_inverse_solve(left, right, var, solve_ops))
  sample = "%s:%s=%s" % (var, left_str, right_str)
  return sample, target


def generate_algebra_simplify_sample(vlist, ops, min_depth, max_depth):
  """Randomly generate an algebra simplify dataset sample.

  Given an input expression, produce the simplified expression.

  See go/symbolic-math-dataset.

  Args:
    vlist: Variable list. List of chars that can be used in the expression.
    ops: List of ExprOp instances. The allowed operators for the expression.
    min_depth: Expression trees will not have a smaller depth than this. 0 means
        there is just a variable. 1 means there is one operation.
    max_depth: Expression trees will not have a larger depth than this. To make
        all trees have the same depth, set this equal to `min_depth`.

  Returns:
    sample: String representation of the input.
    target: String representation of the solution.
  """
  depth = random.randrange(min_depth, max_depth + 1)
  expr = random_expr(depth, vlist, ops)

  sample = str(expr)
  target = format_sympy_expr(sympy.simplify(sample))
  return sample, target


def generate_calculus_integrate_sample(vlist, ops, min_depth, max_depth,
                                       functions):
  """Randomly generate a symbolic integral dataset sample.

  Given an input expression, produce the indefinite integral.

  See go/symbolic-math-dataset.

  Args:
    vlist: Variable list. List of chars that can be used in the expression.
    ops: List of ExprOp instances. The allowed operators for the expression.
    min_depth: Expression trees will not have a smaller depth than this. 0 means
        there is just a variable. 1 means there is one operation.
    max_depth: Expression trees will not have a larger depth than this. To make
        all trees have the same depth, set this equal to `min_depth`.
    functions: Defines special functions. A dict mapping human readable string
        names, like "log", "exp", "sin", "cos", etc., to single chars. Each
        function gets a unique token, like "L" for "log".

  Returns:
    sample: String representation of the input. Will be of the form
        'var:expression'.
    target: String representation of the solution.
  """
  var_index = random.randrange(len(vlist))
  var = vlist[var_index]
  consts = vlist[:var_index] + vlist[var_index + 1:]

  depth = random.randrange(min_depth, max_depth + 1)
  expr = random_expr_with_required_var(depth, var, consts, ops)

  expr_str = str(expr)
  sample = var + ":" + expr_str
  target = format_sympy_expr(
      sympy.integrate(expr_str, sympy.Symbol(var)), functions=functions)
  return sample, target


# AlgebraConfig holds objects required to generate the algebra inverse
# dataset. See go/symbolic-math-dataset.
# vlist: Variable list. A list of chars.
# dlist: Numberical digit list. A list of chars.
# flist: List of special function names. A list of chars.
# functions: Dict of special function names. Maps human readable string names to
#     single char names used in flist.
# ops: Dict mapping op symbols (chars) to ExprOp instances.
# solve_ops: Encodes rules for how to algebraicly cancel out each operation. See
#     doc-string for `algebra_inverse_solve`.
# int_encoder: Function that maps a string to a list of tokens. Use this to
#     encode an expression to feed into a model.
# int_decoder: Function that maps a list of tokens to a string. Use this to
#     convert model input or output into a human readable string.
AlgebraConfig = namedtuple("AlgebraConfig", [
    "vlist", "dlist", "flist", "functions", "ops", "solve_ops", "int_encoder",
    "int_decoder"
])


def math_dataset_init(alphabet_size=26, digits=None, functions=None):
  """Initializes required objects to generate symbolic math datasets.

  See go/symbolic-math-dataset.

  Produces token set, ExprOp instances, solve_op dictionary, encoders, and
  decoders needed to generate the algebra inverse dataset.

  Args:
    alphabet_size: How many possible variables there are. Max 52.
    digits: How many numerical digits to encode as tokens, "0" throuh
        str(digits-1), or None to encode no digits.
    functions: Defines special functions. A dict mapping human readable string
        names, like "log", "exp", "sin", "cos", etc., to single chars. Each
        function gets a unique token, like "L" for "log".
        WARNING, Make sure these tokens do not conflict with the list of
        possible variable names.

  Returns:
    AlgebraConfig instance holding all the objects listed above.

  Raises:
    ValueError: If `alphabet_size` is not in range [2, 52].
  """
  ops_list = ["+", "-", "*", "/"]
  ops = {
      "+": ExprOp("+", 0, True),
      "-": ExprOp("-", 0, False),
      "*": ExprOp("*", 1, True),
      "/": ExprOp("/", 1, False)
  }
  solve_ops = {
      "+l": lambda l, r, to: (l, ExprNode(to, r, ops["-"])),
      "+r": lambda l, r, to: (r, ExprNode(to, l, ops["-"])),
      "-l": lambda l, r, to: (l, ExprNode(to, r, ops["+"])),
      "-r": lambda l, r, to: (r, ExprNode(l, to, ops["-"])),
      "*l": lambda l, r, to: (l, ExprNode(to, r, ops["/"])),
      "*r": lambda l, r, to: (r, ExprNode(to, l, ops["/"])),
      "/l": lambda l, r, to: (l, ExprNode(to, r, ops["*"])),
      "/r": lambda l, r, to: (r, ExprNode(l, to, ops["/"])),
  }
  alphabet = (
      [six.int2byte(ord("a") + c).decode("utf-8") for c in range(26)] +
      [six.int2byte(ord("A") + c).decode("utf-8") for c in range(26)])
  if alphabet_size > 52:
    raise ValueError(
        "alphabet_size cannot be greater than 52. Got %s." % alphabet_size)
  if alphabet_size < 2:
    raise ValueError(
        "alphabet_size cannot be less than 2. Got %s." % alphabet_size)
  if digits is not None and not 1 <= digits <= 10:
    raise ValueError("digits cannot must be between 1 and 10. Got %s." % digits)
  vlist = alphabet[:alphabet_size]
  if digits is not None:
    dlist = [str(d) for d in xrange(digits)]
  else:
    dlist = []
  if functions is None:
    functions = {}
  flist = sorted(functions.values())
  pad = "_"
  tokens = [pad] + [":", "(", ")", "="] + ops_list + vlist + dlist + flist
  if len(tokens) != len(set(tokens)):
    raise ValueError("Duplicate token. Tokens: %s" % tokens)
  token_map = dict([(t, i) for i, t in enumerate(tokens)])

  def int_encoder(sequence):
    return [token_map[s] for s in sequence]

  def int_decoder(tensor_1d):
    return "".join([tokens[i] for i in tensor_1d])

  return AlgebraConfig(
      vlist=vlist,
      dlist=dlist,
      flist=flist,
      functions=functions,
      ops=ops,
      solve_ops=solve_ops,
      int_encoder=int_encoder,
      int_decoder=int_decoder)


def algebra_inverse(alphabet_size=26, min_depth=0, max_depth=2,
                    nbr_cases=10000):
  """Generate the algebra inverse dataset.

  Each sample is a symbolic math equation involving unknown variables. The
  task is to solve for the given variable. The target is the resulting
  expression.

  Args:
    alphabet_size: How many possible variables there are. Max 52.
    min_depth: Minimum depth of the expression trees on both sides of the
        equals sign in the equation.
    max_depth: Maximum depth of the expression trees on both sides of the
        equals sign in the equation.
    nbr_cases: The number of cases to generate.

  Yields:
    A dictionary {"inputs": input-list, "targets": target-list} where
    input-list are the tokens encoding the variable to solve for and the math
    equation, and target-list is a list of tokens encoding the resulting math
    expression after solving for the variable.

  Raises:
    ValueError: If `max_depth` < `min_depth`.
  """

  if max_depth < min_depth:
    raise ValueError("max_depth must be greater than or equal to min_depth. "
                     "Got max_depth=%s, min_depth=%s" % (max_depth, min_depth))

  alg_cfg = math_dataset_init(alphabet_size)
  for _ in xrange(nbr_cases):
    sample, target = generate_algebra_inverse_sample(
        alg_cfg.vlist,
        list(alg_cfg.ops.values()), alg_cfg.solve_ops, min_depth, max_depth)
    yield {
        "inputs": alg_cfg.int_encoder(sample),
        "targets": alg_cfg.int_encoder(target)
    }


def algebra_simplify(alphabet_size=26,
                     min_depth=0,
                     max_depth=2,
                     nbr_cases=10000):
  """Generate the algebra simplify dataset.

  Each sample is a symbolic math expression involving unknown variables. The
  task is to simplify the expression. The target is the resulting expression.

  Args:
    alphabet_size: How many possible variables there are. Max 52.
    min_depth: Minimum depth of the expression trees on both sides of the
        equals sign in the equation.
    max_depth: Maximum depth of the expression trees on both sides of the
        equals sign in the equation.
    nbr_cases: The number of cases to generate.

  Yields:
    A dictionary {"inputs": input-list, "targets": target-list} where
    input-list are the tokens encoding the expression to simplify, and
    target-list is a list of tokens encoding the resulting math expression after
    simplifying.

  Raises:
    ValueError: If `max_depth` < `min_depth`.
  """
  if max_depth < min_depth:
    raise ValueError("max_depth must be greater than or equal to min_depth. "
                     "Got max_depth=%s, min_depth=%s" % (max_depth, min_depth))

  alg_cfg = math_dataset_init(alphabet_size, digits=5)
  for _ in xrange(nbr_cases):
    sample, target = generate_algebra_simplify_sample(
        alg_cfg.vlist, list(alg_cfg.ops.values()), min_depth, max_depth)
    yield {
        "inputs": alg_cfg.int_encoder(sample),
        "targets": alg_cfg.int_encoder(target)
    }


def calculus_integrate(alphabet_size=26,
                       min_depth=0,
                       max_depth=2,
                       nbr_cases=10000):
  """Generate the calculus integrate dataset.

  Each sample is a symbolic math expression involving unknown variables. The
  task is to take the indefinite integral of the expression. The target is the
  resulting expression.

  Args:
    alphabet_size: How many possible variables there are. Max 26.
    min_depth: Minimum depth of the expression trees on both sides of the
        equals sign in the equation.
    max_depth: Maximum depth of the expression trees on both sides of the
        equals sign in the equation.
    nbr_cases: The number of cases to generate.

  Yields:
    A dictionary {"inputs": input-list, "targets": target-list} where
    input-list are the tokens encoding the variable to integrate with respect
    to and the expression to integrate, and target-list is a list of tokens
    encoding the resulting math expression after integrating.

  Raises:
    ValueError: If `max_depth` < `min_depth`, or if alphabet_size > 26.
  """
  if max_depth < min_depth:
    raise ValueError("max_depth must be greater than or equal to min_depth. "
                     "Got max_depth=%s, min_depth=%s" % (max_depth, min_depth))

  # Don't allow alphabet to use capital letters. Those are reserved for function
  # names.
  if alphabet_size > 26:
    raise ValueError(
        "alphabet_size must not be greater than 26. Got %s." % alphabet_size)

  functions = {"log": "L"}
  alg_cfg = math_dataset_init(alphabet_size, digits=5, functions=functions)
  nbr_case = 0
  while nbr_case < nbr_cases:
    try:
      sample, target = generate_calculus_integrate_sample(
          alg_cfg.vlist,
          list(alg_cfg.ops.values()), min_depth, max_depth, alg_cfg.functions)
      yield {
          "inputs": alg_cfg.int_encoder(sample),
          "targets": alg_cfg.int_encoder(target)
      }
    except:  # pylint:disable=bare-except
      continue
    if nbr_case % 10000 == 0:
      print(" calculus_integrate: generating case %d." % nbr_case)
    nbr_case += 1
