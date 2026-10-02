"""Restricted evaluator for computed PsySummary variables."""

import ast


def _attribute_chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


class _ExpressionValidator(ast.NodeVisitor):
    allowed_calls = {"runBoxcox", "np.log", "np.exp", "np.logical_and", "np.logical_or"}
    allowed_attributes = {"self.data", "np.log", "np.exp", "np.logical_and", "np.logical_or"}
    allowed_names = {"self", "np", "runBoxcox"}
    binary_operators = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.BitAnd, ast.BitOr)
    unary_operators = (ast.UAdd, ast.USub, ast.Invert)
    comparison_operators = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)

    def generic_visit(self, node):
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    def visit_Expression(self, node):
        self.visit(node.body)

    def visit_Name(self, node):
        if node.id not in self.allowed_names:
            raise ValueError(f"Unsupported name in expression: {node.id}")

    def visit_Attribute(self, node):
        chain = _attribute_chain(node)
        if chain not in self.allowed_attributes:
            raise ValueError(f"Unsupported attribute access in expression: {chain}")

    def visit_Constant(self, node):
        return None

    def visit_List(self, node):
        for element in node.elts:
            self.visit(element)

    visit_Tuple = visit_List

    def visit_Subscript(self, node):
        self.visit(node.value)
        self.visit(node.slice)

    def visit_Slice(self, node):
        for value in (node.lower, node.upper, node.step):
            if value is not None:
                self.visit(value)

    def visit_BinOp(self, node):
        if not isinstance(node.op, self.binary_operators):
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node):
        if not isinstance(node.op, self.unary_operators):
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        self.visit(node.operand)

    def visit_Compare(self, node):
        self.visit(node.left)
        for operator in node.ops:
            if not isinstance(operator, self.comparison_operators):
                raise ValueError(f"Unsupported comparison: {type(operator).__name__}")
        for comparator in node.comparators:
            self.visit(comparator)

    def visit_Call(self, node):
        call_name = node.func.id if isinstance(node.func, ast.Name) else _attribute_chain(node.func)
        if call_name not in self.allowed_calls:
            raise ValueError(f"Unsupported function in expression: {call_name}")
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            if keyword.arg is None:
                raise ValueError("Unsupported keyword expansion in expression.")
            self.visit(keyword.value)


def evaluate_aggregate_expression(expression, aggregate_data, np_module, boxcox_function):
    """Evaluate a computed-variable expression after validating its syntax."""
    parsed = ast.parse(expression, mode="eval")
    _ExpressionValidator().visit(parsed)
    compiled = compile(parsed, "<aggregate-data-expression>", "eval")
    return eval(
        compiled,
        {"__builtins__": {}},
        {"self": aggregate_data, "np": np_module, "runBoxcox": boxcox_function},
    )
