"""
MechanicsDSL: A Domain-Specific Language for Classical Mechanics

A comprehensive framework for symbolic and numerical analysis of classical 
mechanical systems using LaTeX-inspired notation.

Author: Noah Parsons
Collaboration: Dr. Khalfiah, Stony Brook University
"""

import re
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from typing import List, Dict, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import warnings
from collections import defaultdict

__version__ = "0.1.0"
__author__ = "Noah Parsons"

# ============================================================================
# TOKEN SYSTEM - Enhanced with better coverage
# ============================================================================

TOKEN_TYPES = [
    # Physics specific commands
    ("SYSTEM", r"\\system"),
    ("DEFVAR", r"\\defvar"),
    ("DEFINE", r"\\define"),
    ("LAGRANGIAN", r"\\lagrangian"),
    ("HAMILTONIAN", r"\\hamiltonian"),
    ("TRANSFORM", r"\\transform"),
    ("CONSTRAINT", r"\\constraint"),
    ("INITIAL", r"\\initial"),
    ("SOLVE", r"\\solve"),
    ("ANIMATE", r"\\animate"),
    ("PLOT", r"\\plot"),
    ("PARAMETER", r"\\parameter"),  # NEW: explicit parameter declaration
    
    # Vector operations
    ("VEC", r"\\vec"),
    ("HAT", r"\\hat"),
    ("MAGNITUDE", r"\\mag|\\norm"),
    
    # Time derivatives
    ("DOT_NOTATION", r"\\dot"),
    ("DDOT_NOTATION", r"\\ddot"),
    
    # Advanced math operators
    ("VECTOR_DOT", r"\\cdot"),
    ("VECTOR_CROSS", r"\\times|\\cross"),
    ("GRADIENT", r"\\nabla|\\grad"),
    ("DIVERGENCE", r"\\div"),
    ("CURL", r"\\curl"),
    ("LAPLACIAN", r"\\laplacian|\\Delta"),
    
    # Calculus
    ("PARTIAL", r"\\partial"),
    ("INTEGRAL", r"\\int"),
    ("OINT", r"\\oint"),
    ("SUM", r"\\sum"),
    ("LIMIT", r"\\lim"),
    ("FRAC", r"\\frac"),  # NEW: fraction support
    
    # Greek letters (comprehensive)
    ("GREEK_LETTER", r"\\alpha|\\beta|\\gamma|\\delta|\\epsilon|\\varepsilon|\\zeta|\\eta|\\theta|\\vartheta|\\iota|\\kappa|\\lambda|\\mu|\\nu|\\xi|\\omicron|\\pi|\\varpi|\\rho|\\varrho|\\sigma|\\varsigma|\\tau|\\upsilon|\\phi|\\varphi|\\chi|\\psi|\\omega"),
    
    # General commands
    ("COMMAND", r"\\[a-zA-Z_][a-zA-Z0-9_]*"),
    
    # Brackets and grouping
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    
    # Mathematical operators
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("MULTIPLY", r"\*"),
    ("DIVIDE", r"/"),
    ("POWER", r"\^"),
    ("EQUALS", r"="),
    ("COMMA", r","),
    ("SEMICOLON", r";"),
    ("COLON", r":"),
    ("DOT", r"\."),
    ("UNDERSCORE", r"_"),
    ("PIPE", r"\|"),
    
    # Basic tokens
    ("NUMBER", r"\d+\.?\d*([eE][+-]?\d+)?"),
    ("IDENT", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("WHITESPACE", r"\s+"),
    ("NEWLINE", r"\n"),
    ("COMMENT", r"%.*"),
]

# Compile regex
token_regex = "|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_TYPES)
token_pattern = re.compile(token_regex)

@dataclass
class Token:
    """Token with position tracking for better error messages"""
    type: str
    value: str
    position: int = 0
    line: int = 1
    column: int = 1

    def __repr__(self):
        return f"{self.type}:{self.value}@{self.line}:{self.column}"

def tokenize(source: str) -> List[Token]:
    """
    Tokenizer with position tracking and comprehensive error reporting
    
    Args:
        source: DSL source code
        
    Returns:
        List of tokens (excluding whitespace and comments)
    """
    tokens = []
    line = 1
    line_start = 0
    
    for match in token_pattern.finditer(source):
        kind = match.lastgroup
        value = match.group()
        position = match.start()
        
        # Update line tracking
        while line_start < position and '\n' in source[line_start:position]:
            newline_pos = source.find('\n', line_start)
            if newline_pos != -1 and newline_pos < position:
                line += 1
                line_start = newline_pos + 1
            else:
                break
                
        column = position - line_start + 1
        
        if kind not in ["WHITESPACE", "COMMENT"]:
            tokens.append(Token(kind, value, position, line, column))
            
    return tokens

# ============================================================================
# COMPLETE AST SYSTEM - Enhanced with better type safety
# ============================================================================

class ASTNode:
    """Base class for all AST nodes"""
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class Expression(ASTNode):
    """Base class for all expressions"""
    pass

# Basic expressions
class NumberExpr(Expression):
    def __init__(self, value: float):
        self.value = value
    def __repr__(self):
        return f"Num({self.value})"

class IdentExpr(Expression):
    def __init__(self, name: str):
        self.name = name
    def __repr__(self):
        return f"Id({self.name})"

class GreekLetterExpr(Expression):
    def __init__(self, letter: str):
        self.letter = letter
    def __repr__(self):
        return f"Greek({self.letter})"

class DerivativeVarExpr(Expression):
    """Represents \dot{x} or \ddot{x} notation"""
    def __init__(self, var: str, order: int = 1):
        self.var = var
        self.order = order
    def __repr__(self):
        return f"DerivativeVar({self.var}, order={self.order})"

# Binary operations
class BinaryOpExpr(Expression):
    def __init__(self, left: Expression, operator: str, right: Expression):
        self.left = left
        self.operator = operator
        self.right = right
    def __repr__(self):
        return f"BinOp({self.left} {self.operator} {self.right})"

class UnaryOpExpr(Expression):
    def __init__(self, operator: str, operand: Expression):
        self.operator = operator
        self.operand = operand
    def __repr__(self):
        return f"UnaryOp({self.operator}{self.operand})"

# Vector expressions
class VectorExpr(Expression):
    def __init__(self, components: List[Expression]):
        self.components = components
    def __repr__(self):
        return f"Vector({self.components})"

class VectorOpExpr(Expression):
    def __init__(self, operation: str, left: Expression, right: Expression = None):
        self.operation = operation
        self.left = left
        self.right = right
    def __repr__(self):
        if self.right:
            return f"VectorOp({self.operation}, {self.left}, {self.right})"
        return f"VectorOp({self.operation}, {self.left})"

# Calculus expressions
class DerivativeExpr(Expression):
    def __init__(self, expr: Expression, var: str, order: int = 1, partial: bool = False):
        self.expr = expr
        self.var = var
        self.order = order
        self.partial = partial
    def __repr__(self):
        type_str = "Partial" if self.partial else "Total"
        return f"{type_str}Deriv({self.expr}, {self.var}, order={self.order})"

class IntegralExpr(Expression):
    def __init__(self, expr: Expression, var: str, lower=None, upper=None, line_integral=False):
        self.expr = expr
        self.var = var
        self.lower = lower
        self.upper = upper
        self.line_integral = line_integral
    def __repr__(self):
        return f"Integral({self.expr}, {self.var}, {self.lower}, {self.upper})"

# Function calls
class FunctionCallExpr(Expression):
    def __init__(self, name: str, args: List[Expression]):
        self.name = name
        self.args = args
    def __repr__(self):
        return f"Call({self.name}, {self.args})"

# NEW: Fraction expression
class FractionExpr(Expression):
    def __init__(self, numerator: Expression, denominator: Expression):
        self.numerator = numerator
        self.denominator = denominator
    def __repr__(self):
        return f"Frac({self.numerator}/{self.denominator})"

# Physics-specific AST nodes
class SystemDef(ASTNode):
    def __init__(self, name: str):
        self.name = name
    def __repr__(self):
        return f"System({self.name})"

class VarDef(ASTNode):
    def __init__(self, name: str, vartype: str, unit: str, vector: bool = False):
        self.name = name
        self.vartype = vartype
        self.unit = unit
        self.vector = vector
    def __repr__(self):
        vec_str = " [Vector]" if self.vector else ""
        return f"VarDef({self.name}: {self.vartype}[{self.unit}]{vec_str})"

# NEW: Parameter definition
class ParameterDef(ASTNode):
    def __init__(self, name: str, value: float, unit: str):
        self.name = name
        self.value = value
        self.unit = unit
    def __repr__(self):
        return f"Parameter({self.name} = {self.value} [{self.unit}])"

class DefineDef(ASTNode):
    def __init__(self, name: str, args: List[str], body: Expression):
        self.name = name
        self.args = args
        self.body = body
    def __repr__(self):
        return f"Define({self.name}({', '.join(self.args)}) = {self.body})"

class LagrangianDef(ASTNode):
    def __init__(self, expr: Expression):
        self.expr = expr
    def __repr__(self):
        return f"Lagrangian({self.expr})"

class HamiltonianDef(ASTNode):
    def __init__(self, expr: Expression):
        self.expr = expr
    def __repr__(self):
        return f"Hamiltonian({self.expr})"

class TransformDef(ASTNode):
    def __init__(self, coord_type: str, var: str, expr: Expression):
        self.coord_type = coord_type
        self.var = var
        self.expr = expr
    def __repr__(self):
        return f"Transform({self.coord_type}: {self.var} = {self.expr})"

# NEW: Constraint definition
class ConstraintDef(ASTNode):
    def __init__(self, expr: Expression, constraint_type: str = "holonomic"):
        self.expr = expr
        self.constraint_type = constraint_type
    def __repr__(self):
        return f"Constraint({self.expr}, type={self.constraint_type})"

class InitialCondition(ASTNode):
    def __init__(self, conditions: Dict[str, float]):
        self.conditions = conditions
    def __repr__(self):
        return f"Initial({self.conditions})"

class SolveDef(ASTNode):
    def __init__(self, method: str, options: Dict[str, Any] = None):
        self.method = method
        self.options = options or {}
    def __repr__(self):
        return f"Solve({self.method}, {self.options})"

class AnimateDef(ASTNode):
    def __init__(self, target: str, options: Dict[str, Any] = None):
        self.target = target
        self.options = options or {}
    def __repr__(self):
        return f"Animate({self.target}, {self.options})"

# ============================================================================
# COMPREHENSIVE PHYSICS UNITS SYSTEM
# ============================================================================

@dataclass
class Unit:
    """Physical unit with dimensional analysis"""
    dimensions: Dict[str, int] = field(default_factory=dict)
    scale: float = 1.0

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Unit(self.dimensions.copy(), self.scale * other)
        result = {}
        all_dims = set(self.dimensions.keys()) | set(other.dimensions.keys())
        for dim in all_dims:
            result[dim] = self.dimensions.get(dim, 0) + other.dimensions.get(dim, 0)
            if result[dim] == 0:
                del result[dim]
        return Unit(result, self.scale * other.scale)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Unit(self.dimensions.copy(), self.scale / other)
        result = {}
        all_dims = set(self.dimensions.keys()) | set(other.dimensions.keys())
        for dim in all_dims:
            result[dim] = self.dimensions.get(dim, 0) - other.dimensions.get(dim, 0)
            if result[dim] == 0:
                del result[dim]
        return Unit(result, self.scale / other.scale)

    def __pow__(self, exponent):
        result = {dim: power * exponent for dim, power in self.dimensions.items()}
        return Unit(result, self.scale ** exponent)

    def is_compatible(self, other):
        """Check if units are dimensionally compatible"""
        return self.dimensions == other.dimensions

    def __repr__(self):
        if not self.dimensions:
            return f"Unit(dimensionless, scale={self.scale})"
        return f"Unit({self.dimensions}, scale={self.scale})"

# Comprehensive unit system
BASE_UNITS = {
    "dimensionless": Unit({}),
    "1": Unit({}),
    
    # SI Base units
    "m": Unit({"length": 1}),
    "kg": Unit({"mass": 1}),
    "s": Unit({"time": 1}),
    "A": Unit({"current": 1}),
    "K": Unit({"temperature": 1}),
    "mol": Unit({"substance": 1}),
    "cd": Unit({"luminous_intensity": 1}),
    
    # Common derived units
    "N": Unit({"mass": 1, "length": 1, "time": -2}),  # Force
    "J": Unit({"mass": 1, "length": 2, "time": -2}),  # Energy
    "W": Unit({"mass": 1, "length": 2, "time": -3}),  # Power
    "Pa": Unit({"mass": 1, "length": -1, "time": -2}), # Pressure
    "Hz": Unit({"time": -1}),  # Frequency
    "C": Unit({"current": 1, "time": 1}),  # Charge
    "V": Unit({"mass": 1, "length": 2, "time": -3, "current": -1}),  # Voltage
    "F": Unit({"mass": -1, "length": -2, "time": 4, "current": 2}),  # Capacitance
    "Wb": Unit({"mass": 1, "length": 2, "time": -2, "current": -1}),  # Magnetic flux
    "T": Unit({"mass": 1, "time": -2, "current": -1}),  # Magnetic field
    
    # Angle units
    "rad": Unit({"angle": 1}),
    "deg": Unit({"angle": 1}, scale=np.pi/180),
    
    # Common physics constants (as units)
    "c": Unit({"length": 1, "time": -1}, scale=299792458),  # Speed of light
    "hbar": Unit({"mass": 1, "length": 2, "time": -1}, scale=1.055e-34),
    "G": Unit({"mass": -1, "length": 3, "time": -2}, scale=6.674e-11),
    "k_B": Unit({"mass": 1, "length": 2, "time": -2, "temperature": -1}, scale=1.381e-23),
}

class UnitSystem:
    """Manages unit operations and conversions"""
    
    def __init__(self):
        self.units = BASE_UNITS.copy()
        
    def parse_unit(self, unit_str: str) -> Unit:
        """Parse unit string like 'kg*m/s^2' into Unit object"""
        if unit_str in self.units:
            return self.units[unit_str]
        
        # Try to parse compound units
        # This is simplified - full implementation would need proper parsing
        try:
            # Handle basic cases
            if '*' in unit_str or '/' in unit_str or '^' in unit_str:
                # Use eval in a controlled namespace (security consideration for production)
                namespace = {k: v for k, v in self.units.items()}
                return eval(unit_str, {"__builtins__": {}}, namespace)
            return Unit({})  # Fallback to dimensionless
        except:
            warnings.warn(f"Could not parse unit: {unit_str}")
            return Unit({})
    
    def check_compatibility(self, unit1: str, unit2: str) -> bool:
        """Check if two units are compatible"""
        u1 = self.parse_unit(unit1)
        u2 = self.parse_unit(unit2)
        return u1.is_compatible(u2)

# ============================================================================
# ENHANCED PARSER ENGINE
# ============================================================================

class MechanicsParser:
    """Parser with improved error handling and feature completeness"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_system = None
        self.errors = []

    def peek(self, offset: int = 0) -> Optional[Token]:
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None

    def match(self, *expected_types: str) -> Optional[Token]:
        token = self.peek()
        if token and token.type in expected_types:
            self.pos += 1
            return token
        return None

    def expect(self, expected_type: str) -> Token:
        token = self.match(expected_type)
        if not token:
            current = self.peek()
            if current:
                error_msg = f"Expected {expected_type} but got {current.type} '{current.value}' at {current.line}:{current.column}"
                self.expect("RPAREN")
                
                if isinstance(expr, IdentExpr):
                    expr = FunctionCallExpr(expr.name, args)
                else:
                    raise SyntaxError("Invalid function call syntax")
                    
            else:
                break
                
        return expr

    def parse_primary(self) -> Expression:
        """Primary expressions: literals, identifiers, parentheses, vectors, commands"""

        # Numbers
        token = self.peek()
        if self.match("NUMBER"):
            return NumberExpr(float(self.tokens[self.pos - 1].value))

        # Time derivatives: \dot{x} and \ddot{x}
        if self.match("DOT_NOTATION"):
            self.expect("LBRACE")
            var = self.expect("IDENT").value
            self.expect("RBRACE")
            return DerivativeVarExpr(var, 1)
            
        if self.match("DDOT_NOTATION"):
            self.expect("LBRACE")
            var = self.expect("IDENT").value
            self.expect("RBRACE")
            return DerivativeVarExpr(var, 2)

        # Identifiers
        if self.match("IDENT"):
            return IdentExpr(self.tokens[self.pos - 1].value)

        # Greek letters
        if self.match("GREEK_LETTER"):
            letter = self.tokens[self.pos - 1].value[1:]  # Remove backslash
            return GreekLetterExpr(letter)

        # Parentheses
        if self.match("LPAREN"):
            expr = self.parse_expression()
            self.expect("RPAREN")
            return expr

        # Vectors [x, y, z]
        if self.match("LBRACKET"):
            components = []
            components.append(self.parse_expression())
            while self.match("COMMA"):
                components.append(self.parse_expression())
            self.expect("RBRACKET")
            return VectorExpr(components)

        # Commands (LaTeX-style functions)
        if self.match("COMMAND"):
            cmd = self.tokens[self.pos - 1].value
            return self.parse_command(cmd)

        # Mathematical constants
        if token and token.value in ["pi", "e"]:
            self.pos += 1
            if token.value == "pi":
                return NumberExpr(np.pi)
            elif token.value == "e":
                return NumberExpr(np.e)

        current = self.peek()
        if current:
            raise SyntaxError(f"Unexpected token {current.type} '{current.value}' at {current.line}:{current.column}")
        else:
            raise SyntaxError("Unexpected end of input")

    def parse_command(self, cmd: str) -> Expression:
        """Parse LaTeX-style commands"""
        
        if cmd == r"\frac":
            # \frac{numerator}{denominator}
            self.expect("LBRACE")
            num = self.parse_expression()
            self.expect("RBRACE")
            self.expect("LBRACE")
            denom = self.parse_expression()
            self.expect("RBRACE")
            return FractionExpr(num, denom)
        
        elif cmd == r"\vec":
            self.expect("LBRACE")
            expr = self.parse_expression()
            self.expect("RBRACE")
            return VectorOpExpr("vec", expr)
            
        elif cmd == r"\hat":
            self.expect("LBRACE")
            expr = self.parse_expression()
            self.expect("RBRACE")
            return VectorOpExpr("unit", expr)
            
        elif cmd in [r"\mag", r"\norm"]:
            self.expect("LBRACE")
            expr = self.parse_expression()
            self.expect("RBRACE")
            return VectorOpExpr("magnitude", expr)
            
        elif cmd == r"\partial":
            # \partial{f}{x}
            self.expect("LBRACE")
            expr = self.parse_expression()
            self.expect("RBRACE")
            self.expect("LBRACE")
            var = self.expect("IDENT").value
            self.expect("RBRACE")
            return DerivativeExpr(expr, var, 1, True)
            
        elif cmd in [r"\sin", r"\cos", r"\tan", r"\exp", r"\log", r"\ln", r"\sqrt", 
                     r"\sinh", r"\cosh", r"\tanh", r"\arcsin", r"\arccos", r"\arctan"]:
            func_name = cmd[1:]  # Remove backslash
            self.expect("LBRACE")
            arg = self.parse_expression()
            self.expect("RBRACE")
            return FunctionCallExpr(func_name, [arg])
            
        elif cmd in [r"\nabla", r"\grad"]:
            # Gradient operator
            if self.peek() and self.peek().type == "LBRACE":
                self.expect("LBRACE")
                expr = self.parse_expression()
                self.expect("RBRACE")
                return VectorOpExpr("grad", expr)
            return VectorOpExpr("grad", None)
            
        else:
            # Unknown command - treat as identifier
            return IdentExpr(cmd[1:])

    def at_end_of_expression(self) -> bool:
        """Check if we're at the end of an expression"""
        token = self.peek()
        return (not token or 
                token.type in ["RBRACE", "RPAREN", "RBRACKET", "COMMA", "SEMICOLON", "EQUALS"])

    def expression_to_string(self, expr: Expression) -> str:
        """Convert expression back to string for unit parsing"""
        if isinstance(expr, NumberExpr):
            return str(expr.value)
        elif isinstance(expr, IdentExpr):
            return expr.name
        elif isinstance(expr, BinaryOpExpr):
            left = self.expression_to_string(expr.left)
            right = self.expression_to_string(expr.right)
            return f"({left} {expr.operator} {right})"
        elif isinstance(expr, UnaryOpExpr):
            operand = self.expression_to_string(expr.operand)
            return f"{expr.operator}{operand}"
        else:
            return str(expr)

# ============================================================================
# ENHANCED SYMBOLIC MATH ENGINE
# ============================================================================

class SymbolicEngine:
    """
    Enhanced symbolic mathematics engine with better error handling
    and support for all DSL features
    """
    
    def __init__(self):
        self.sp = sp
        self.symbol_map = {}
        self.function_map = {}
        self.time_symbol = sp.Symbol('t', real=True)
        self.assumptions = {}  # Track assumptions about symbols

    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a SymPy symbol with assumptions"""
        if name not in self.symbol_map:
            # Default assumptions for physical quantities
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self.symbol_map[name] = sp.Symbol(name, **default_assumptions)
            self.assumptions[name] = default_assumptions
        return self.symbol_map[name]

    def get_function(self, name: str) -> sp.Function:
        """Get or create a SymPy function"""
        if name not in self.function_map:
            self.function_map[name] = sp.Function(name, real=True)
        return self.function_map[name]

    def ast_to_sympy(self, expr: Expression) -> sp.Expr:
        """
        Convert AST expression to SymPy with comprehensive support
        
        Args:
            expr: AST expression node
            
        Returns:
            SymPy expression
            
        Raises:
            ValueError: If expression type is not supported
        """
        
        if isinstance(expr, NumberExpr):
            return sp.Float(expr.value)
            
        elif isinstance(expr, IdentExpr):
            return self.get_symbol(expr.name)
            
        elif isinstance(expr, GreekLetterExpr):
            return self.get_symbol(expr.letter)
            
        elif isinstance(expr, BinaryOpExpr):
            left = self.ast_to_sympy(expr.left)
            right = self.ast_to_sympy(expr.right)
            
            ops = {
                "+": lambda l, r: l + r,
                "-": lambda l, r: l - r,
                "*": lambda l, r: l * r,
                "/": lambda l, r: l / r,
                "^": lambda l, r: l ** r,
            }
            
            if expr.operator in ops:
                return ops[expr.operator](left, right)
            else:
                raise ValueError(f"Unknown operator: {expr.operator}")
                
        elif isinstance(expr, UnaryOpExpr):
            operand = self.ast_to_sympy(expr.operand)
            if expr.operator == "-":
                return -operand
            elif expr.operator == "+":
                return operand
            else:
                raise ValueError(f"Unknown unary operator: {expr.operator}")
        
        elif isinstance(expr, FractionExpr):
            num = self.ast_to_sympy(expr.numerator)
            denom = self.ast_to_sympy(expr.denominator)
            return num / denom

        elif isinstance(expr, DerivativeVarExpr):
            # Convert \dot{x} to x_dot symbol
            if expr.order == 1:
                return self.get_symbol(f"{expr.var}_dot")
            elif expr.order == 2:
                return self.get_symbol(f"{expr.var}_ddot")
            else:
                raise ValueError(f"Derivative order {expr.order} not supported")
                
        elif isinstance(expr, DerivativeExpr):
            inner = self.ast_to_sympy(expr.expr)
            var = self.get_symbol(expr.var)
            
            if expr.partial:
                return sp.diff(inner, var, expr.order)
            else:
                # Time derivative
                if expr.var == "t":
                    return sp.diff(inner, self.time_symbol, expr.order)
                else:
                    return sp.diff(inner, var, expr.order)
                    
        elif isinstance(expr, FunctionCallExpr):
            args = [self.ast_to_sympy(arg) for arg in expr.args]
            
            # Built-in functions
            builtin_funcs = {
                "sin": sp.sin,
                "cos": sp.cos,
                "tan": sp.tan,
                "exp": sp.exp,
                "log": sp.log,
                "ln": sp.log,
                "sqrt": sp.sqrt,
                "sinh": sp.sinh,
                "cosh": sp.cosh,
                "tanh": sp.tanh,
                "arcsin": sp.asin,
                "arccos": sp.acos,
                "arctan": sp.atan,
                "abs": sp.Abs,
            }
            
            if expr.name in builtin_funcs:
                return builtin_funcs[expr.name](*args)
            elif expr.name == "dot":
                # Vector dot product
                if len(args) % 2 == 0:
                    n = len(args) // 2
                    return sum(args[i] * args[i+n] for i in range(n))
                else:
                    raise ValueError("Dot product requires even number of arguments")
            else:
                # Custom function
                func = self.get_function(expr.name)
                return func(*args)
                
        elif isinstance(expr, VectorExpr):
            # Convert to list of sympy expressions
            return sp.Matrix([self.ast_to_sympy(comp) for comp in expr.components])
            
        elif isinstance(expr, VectorOpExpr):
            if expr.operation == "grad":
                # Gradient operation
                if expr.left:
                    inner = self.ast_to_sympy(expr.left)
                    # Return gradient as Matrix
                    vars_list = [self.get_symbol(v) for v in ['x', 'y', 'z']]
                    return sp.Matrix([sp.diff(inner, var) for var in vars_list])
                else:
                    return self.get_symbol('nabla')
            elif expr.operation == "dot":
                left_vec = self.ast_to_sympy(expr.left)
                right_vec = self.ast_to_sympy(expr.right)
                if isinstance(left_vec, sp.Matrix) and isinstance(right_vec, sp.Matrix):
                    return left_vec.dot(right_vec)
                else:
                    return left_vec * right_vec
            elif expr.operation == "cross":
                # Cross product
                left_vec = self.ast_to_sympy(expr.left)
                right_vec = self.ast_to_sympy(expr.right)
                if isinstance(left_vec, sp.Matrix) and isinstance(right_vec, sp.Matrix):
                    return left_vec.cross(right_vec)
                else:
                    raise ValueError("Cross product requires vector arguments")
            elif expr.operation == "magnitude":
                vec = self.ast_to_sympy(expr.left)
                if isinstance(vec, sp.Matrix):
                    return sp.sqrt(vec.dot(vec))
                else:
                    return sp.Abs(vec)
                    
        else:
            raise ValueError(f"Cannot convert {type(expr).__name__} to SymPy")

    def derive_equations_of_motion(self, lagrangian: sp.Expr, coordinates: List[str]) -> List[sp.Expr]:
        """
        Derive Euler-Lagrange equations from Lagrangian
        
        The Euler-Lagrange equation is:
        d/dt(∂L/∂q̇) - ∂L/∂q = 0
        
        Args:
            lagrangian: Lagrangian expression
            coordinates: List of generalized coordinates
            
        Returns:
            List of equations of motion (one per coordinate)
        """
        equations = []
        
        for q in coordinates:
            q_sym = self.get_symbol(q)
            q_dot_sym = self.get_symbol(f"{q}_dot")
            q_ddot_sym = self.get_symbol(f"{q}_ddot")

            # Create time-dependent function for proper differentiation
            q_func = sp.Function(q)(self.time_symbol)

            # Substitute coordinate and velocity with time-dependent functions
            L_with_funcs = lagrangian.subs(q_sym, q_func)
            L_with_funcs = L_with_funcs.subs(q_dot_sym, sp.diff(q_func, self.time_symbol))

            # ∂L/∂q̇
            dL_dq_dot = sp.diff(L_with_funcs, sp.diff(q_func, self.time_symbol))

            # d/dt(∂L/∂q̇)
            d_dt_dL_dq_dot = sp.diff(dL_dq_dot, self.time_symbol)

            # ∂L/∂q
            dL_dq = sp.diff(L_with_funcs, q_func)

            # Euler-Lagrange equation: d/dt(∂L/∂q̇) - ∂L/∂q = 0
            equation = d_dt_dL_dq_dot - dL_dq

            # Substitute back to symbolic form
            equation = equation.subs(q_func, q_sym)
            equation = equation.subs(sp.diff(q_func, self.time_symbol), q_dot_sym)
            equation = equation.subs(sp.diff(q_func, self.time_symbol, 2), q_ddot_sym)

            # Simplify
            equation = sp.simplify(equation)
            equations.append(equation)
            
        return equations

    def solve_for_accelerations(self, equations: List[sp.Expr], coordinates: List[str]) -> Dict[str, sp.Expr]:
        """
        Solve equations of motion for accelerations (q̈)
        
        Args:
            equations: List of equations of motion
            coordinates: List of generalized coordinates
            
        Returns:
            Dictionary mapping acceleration symbols to their expressions
        """
        accelerations = {}
        accel_symbols = [self.get_symbol(f"{q}_ddot") for q in coordinates]
        
        try:
            # Try to solve the entire system at once
            solutions = sp.solve(equations, accel_symbols, dict=True)
            
            if solutions:
                # Take first solution
                sol = solutions[0] if isinstance(solutions, list) else solutions
                for q in coordinates:
                    accel_key = f"{q}_ddot"
                    if accel_key in sol:
                        accelerations[accel_key] = sp.simplify(sol[accel_key])
                    elif self.get_symbol(accel_key) in sol:
                        accelerations[accel_key] = sp.simplify(sol[self.get_symbol(accel_key)])
            else:
                # Fall back to individual equation solving
                for i, q in enumerate(coordinates):
                    accel_key = f"{q}_ddot"
                    try:
                        sol = sp.solve(equations[i], accel_symbols[i])
                        if sol:
                            accelerations[accel_key] = sp.simplify(sol[0] if isinstance(sol, list) else sol)
                    except Exception as e:
                        warnings.warn(f"Could not solve for {accel_key}: {e}")
                        accelerations[accel_key] = equations[i]
                        
        except Exception as e:
            warnings.warn(f"Could not solve equations symbolically: {e}")
            # Return unsolved equations
            for i, q in enumerate(coordinates):
                accelerations[f"{q}_ddot"] = equations[i]
                
        return accelerations
    
    def calculate_energy(self, lagrangian: sp.Expr, coordinates: List[str], 
                        parameters: Dict[str, float]) -> Tuple[sp.Expr, sp.Expr]:
        """
        Calculate kinetic and potential energy from Lagrangian
        L = T - V, where T is kinetic energy and V is potential energy
        
        Args:
            lagrangian: Lagrangian expression
            coordinates: List of generalized coordinates
            parameters: Physical parameters
            
        Returns:
            Tuple of (kinetic_energy, potential_energy)
        """
        # This is a simplified approach - assumes L = T - V structure
        # A more sophisticated version would use the Hamiltonian formulation
        
        # Substitute parameters
        L = lagrangian
        for param, value in parameters.items():
            L = L.subs(self.get_symbol(param), value)
        
        # Separate velocity-dependent and position-dependent terms
        velocity_terms = sp.S.Zero
        position_terms = sp.S.Zero
        
        for q in coordinates:
            q_dot = self.get_symbol(f"{q}_dot")
            # Extract terms with q_dot (kinetic energy)
            for term in sp.Add.make_args(L):
                if term.has(q_dot):
                    velocity_terms += term
                else:
                    position_terms += term
        
        T = velocity_terms  # Kinetic energy
        V = -position_terms  # Potential energy (note the sign from L = T - V)
        
        return sp.simplify(T), sp.simplify(V)

# ============================================================================
# ENHANCED NUMERICAL SIMULATION ENGINE
# ============================================================================

class NumericalSimulator:
    """
    Enhanced numerical simulator with better stability and error handling
    """
    
    def __init__(self, symbolic_engine: SymbolicEngine):
        self.symbolic = symbolic_engine
        self.equations = {}
        self.parameters = {}
        self.initial_conditions = {}
        self.constraints = []
        self.state_vars = []
        self.coordinates = []

    def set_parameters(self, params: Dict[str, float]):
        """Set physical parameters"""
        self.parameters.update(params)

    def set_initial_conditions(self, conditions: Dict[str, float]):
        """Set initial conditions"""
        self.initial_conditions.update(conditions)
    
    def add_constraint(self, constraint_expr: sp.Expr):
        """Add a constraint equation"""
        self.constraints.append(constraint_expr)

    def compile_equations(self, accelerations: Dict[str, sp.Expr], coordinates: List[str]):
        """
        Compile symbolic equations to numerical functions with improved robustness
        
        Args:
            accelerations: Dictionary of acceleration expressions
            coordinates: List of generalized coordinates
        """
        
        # Create state vector: [q1, q1_dot, q2, q2_dot, ...]
        state_vars = []
        for q in coordinates:
            state_vars.extend([q, f"{q}_dot"])
            
        # Create parameter substitutions
        param_subs = {self.symbolic.get_symbol(k): v for k, v in self.parameters.items()}
        
        # Compile each acceleration equation
        compiled_equations = {}
        
        for q in coordinates:
            accel_key = f"{q}_ddot"
            if accel_key in accelerations:
                # Substitute parameters
                eq = accelerations[accel_key].subs(param_subs)
                
                # Simplify and expand
                try:
                    eq = sp.simplify(eq)
                    eq = sp.expand(eq)
                except:
                    pass

                # Replace any remaining Derivative objects with symbols
                eq = self._replace_derivatives(eq, coordinates)
                
                # Get free symbols and their indices in state vector
                free_symbols = eq.free_symbols
                ordered_symbols = []
                symbol_indices = []
                
                for i, var_name in enumerate(state_vars):
                    sym = self.symbolic.get_symbol(var_name)
                    if sym in free_symbols:
                        ordered_symbols.append(sym)
                        symbol_indices.append(i)
                
                # Compile to numerical function
                if ordered_symbols:
                    try:
                        # Use lambdify with numpy for better performance
                        func = sp.lambdify(ordered_symbols, eq, modules=['numpy', 'math'])
                        
                        def make_wrapper(func, indices):
                            def wrapper(*state_vector):
                                try:
                                    args = [state_vector[i] for i in indices if i < len(state_vector)]
                                    if len(args) == len(indices):
                                        result = func(*args)
                                        # Handle array results
                                        if isinstance(result, np.ndarray):
                                            result = float(result.item()) if result.size == 1 else float(result[0])
                                        result = float(result)
                                        return result if np.isfinite(result) else 0.0
                                    return 0.0
                                except Exception as e:
                                    warnings.warn(f"Evaluation error in {accel_key}: {e}")
                                    return 0.0
                            return wrapper
                        
                        compiled_equations[accel_key] = make_wrapper(func, symbol_indices)
                        
                    except Exception as e:
                        warnings.warn(f"Compilation failed for {accel_key}: {e}")
                        compiled_equations[accel_key] = lambda *args: 0.0
                else:
                    # Constant expression
                    try:
                        const_value = float(sp.N(eq))
                        compiled_equations[accel_key] = lambda *args: const_value
                    except:
                        compiled_equations[accel_key] = lambda *args: 0.0

        self.equations = compiled_equations
        self.state_vars = state_vars
        self.coordinates = coordinates

    def _replace_derivatives(self, expr: sp.Expr, coordinates: List[str]) -> sp.Expr:
        """Replace Derivative objects with corresponding symbols"""
        derivs = list(expr.atoms(sp.Derivative))
        for d in derivs:
            try:
                base = d.args[0]
                # Determine order
                order = 1
                if len(d.args) >= 2:
                    arg2 = d.args[1]
                    if isinstance(arg2, tuple) and len(arg2) >= 2:
                        order = int(arg2[1])
                    else:
                        order = 1
                
                base_name = str(base)
                if base_name in coordinates:
                    if order == 1:
                        repl = self.symbolic.get_symbol(f"{base_name}_dot")
                    elif order == 2:
                        repl = self.symbolic.get_symbol(f"{base_name}_ddot")
                    else:
                        continue
                    expr = expr.xreplace({d: repl})
            except Exception:
                continue
        return expr

    def equations_of_motion(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        ODE system for numerical integration with error handling
        
        Args:
            t: Current time
            y: State vector [q1, q1_dot, q2, q2_dot, ...]
            
        Returns:
            Derivative of state vector
        """
        
        dydt = np.zeros_like(y)
        
        # First half: derivatives of positions are velocities
        for i in range(len(self.coordinates)):
            if 2*i + 1 < len(y):
                dydt[2*i] = y[2*i + 1]
        
        # Second half: derivatives of velocities are accelerations
        for i, q in enumerate(self.coordinates):
            accel_key = f"{q}_ddot"
            if accel_key in self.equations and 2*i + 1 < len(dydt):
                try:
                    accel_value = self.equations[accel_key](*y)
                    if np.isfinite(accel_value):
                        dydt[2*i + 1] = accel_value
                    else:
                        dydt[2*i + 1] = 0.0
                except Exception as e:
                    warnings.warn(f"Error evaluating {accel_key} at t={t}: {e}")
                    dydt[2*i + 1] = 0.0
                    
        return dydt

    def simulate(self, t_span: Tuple[float, float], num_points: int = 1000,
                 method: str = 'RK45', rtol: float = 1e-6, atol: float = 1e-8) -> dict:
        """
        Run numerical simulation with adaptive integration
        
        Args:
            t_span: Time span (t_start, t_end)
            num_points: Number of output points
            method: Integration method ('RK45', 'DOP853', 'LSODA', etc.)
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            Dictionary with solution data and metadata
        """
        
        # Set up initial conditions vector
        y0 = []
        for q in self.coordinates:
            pos_val = self.initial_conditions.get(q, 0.0)
            y0.append(pos_val)
            vel_key = f"{q}_dot"
            vel_val = self.initial_conditions.get(vel_key, 0.0)
            y0.append(vel_val)

        y0 = np.array(y0, dtype=float)

        # Time points
        t_eval = np.linspace(t_span[0], t_span[1], num_points)

        # Validate initial conditions
        try:
            dydt_test = self.equations_of_motion(t_span[0], y0)
            if not np.all(np.isfinite(dydt_test)):
                return {
                    'success': False,
                    'error': 'Initial derivatives contain non-finite values'
                }
        except Exception as e:
            return {
                'success': False,
                'error': f'Initial evaluation failed: {str(e)}'
            }
        
        # Solve ODE
        try:
            solution = solve_ivp(
                self.equations_of_motion,
                t_span,
                y0,
                t_eval=t_eval,
                method=method,
                rtol=rtol,
                atol=atol,
                max_step=min(0.01, (t_span[1] - t_span[0]) / 100)
            )
            
            return {
                'success': solution.success,
                't': solution.t,
                'y': solution.y,
                'coordinates': self.coordinates,
                'state_vars': self.state_vars,
                'message': solution.message if hasattr(solution, 'message') else '',
                'nfev': solution.nfev if hasattr(solution, 'nfev') else 0,
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# ============================================================================
# ENHANCED 3D VISUALIZATION ENGINE
# ============================================================================

class MechanicsVisualizer:
    """Enhanced visualization with more system types and better animations"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        self.animation = None

    def has_ffmpeg(self) -> bool:
        """Check if ffmpeg is available for video export"""
        import shutil
        return shutil.which('ffmpeg') is not None

    def save_animation_to_file(self, anim: animation.FuncAnimation, 
                               filename: str, fps: int = 30, dpi: int = 100) -> bool:
        """
        Save animation to file (MP4 or GIF)
        
        Args:
            anim: Matplotlib animation object
            filename: Output filename
            fps: Frames per second
            dpi: Resolution
            
        Returns:
            True if successful
        """
        if anim is None:
            return False

        try:
            if filename.lower().endswith('.mp4') and self.has_ffmpeg():
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=fps, metadata=dict(artist='MechanicsDSL'), bitrate=1800)
                anim.save(filename, writer=writer, dpi=dpi)
                return True
            elif filename.lower().endswith('.gif'):
                anim.save(filename, writer='pillow', fps=fps)
                return True
            else:
                # Try MP4 anyway
                if self.has_ffmpeg():
                    Writer = animation.writers['ffmpeg']
                    writer = Writer(fps=fps, metadata=dict(artist='MechanicsDSL'), bitrate=1800)
                    anim.save(filename, writer=writer, dpi=dpi)
                    return True
        except Exception as e:
            warnings.warn(f"Failed to save animation: {e}")
        
        return False

    def setup_3d_plot(self, title: str = "Classical Mechanics Simulation"):
        """Setup 3D plotting environment"""
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Y (m)', fontsize=12)
        self.ax.set_zlabel('Z (m)', fontsize=12)
        self.ax.grid(True, alpha=0.3)

    def animate_pendulum(self, solution: dict, parameters: dict, system_name: str = "pendulum"):
        """Create animated pendulum visualization (single or double)"""
        
        if not solution['success']:
            warnings.warn("Cannot animate failed simulation")
            return None
            
        self.setup_3d_plot(f"{system_name.title()} Animation")
        
        t = solution['t']
        y = solution['y']
        coordinates = solution['coordinates']
        
        name = (system_name or '').lower()
        
        # Determine if single or double pendulum
        if len(coordinates) >= 2 or 'double' in name:
            return self._animate_double_pendulum(t, y, parameters)
        else:
            return self._animate_single_pendulum(t, y, parameters)
    
    def _animate_single_pendulum(self, t: np.ndarray, y: np.ndarray, parameters: dict):
        """Animate single pendulum"""
        theta = y[0]
        l = parameters.get('l', 1.0)
        
        # Convert to Cartesian
        x = l * np.sin(theta)
        y_pos = -l * np.cos(theta)
        z = np.zeros_like(x)
        
        # Set axis limits
        self.ax.set_xlim(-l*1.2, l*1.2)
        self.ax.set_ylim(-l*1.2, l*0.2)
        self.ax.set_zlim(-0.1, 0.1)
        
        # Plot elements
        line, = self.ax.plot([], [], [], 'o-', linewidth=3, markersize=10, 
                            color='#E63946', label='Pendulum')
        trail, = self.ax.plot([], [], [], '-', alpha=0.4, linewidth=1.5, 
                             color='#457B9D', label='Trail')
        time_text = self.ax.text2D(0.05, 0.95, '', transform=self.ax.transAxes, fontsize=12)
        
        self.ax.legend(loc='upper right')
        
        def animate_frame(frame):
            if frame < len(t):
                # Pendulum rod and bob
                line.set_data([0, x[frame]], [0, y_pos[frame]])
                line.set_3d_properties([0, z[frame]])
                
                # Trail
                trail_length = min(frame, 150)
                if trail_length > 0:
                    trail.set_data(x[frame-trail_length:frame+1], 
                                 y_pos[frame-trail_length:frame+1])
                    trail.set_3d_properties(z[frame-trail_length:frame+1])
                
                time_text.set_text(f'Time: {t[frame]:.2f} s')
                
            return line, trail, time_text
        
        interval = max(1, int(1000 * (t[-1] - t[0]) / len(t)))
        self.animation = animation.FuncAnimation(
            self.fig, animate_frame, frames=len(t),
            interval=interval, blit=False, repeat=True
        )
        
        return self.animation
    
    def _animate_double_pendulum(self, t: np.ndarray, y: np.ndarray, parameters: dict):
        """Animate double pendulum"""
        theta1 = y[0]
        theta2 = y[2] if y.shape[0] >= 4 else y[0]
        
        l1 = parameters.get('l1', parameters.get('l', 1.0))
        l2 = parameters.get('l2', 1.0)
        
        # Positions
        x1 = l1 * np.sin(theta1)
        y1 = -l1 * np.cos(theta1)
        x2 = x1 + l2 * np.sin(theta2)
        y2 = y1 - l2 * np.cos(theta2)
        
        # Set axis limits
        max_reach = l1 + l2
        self.ax.set_xlim(-max_reach*1.1, max_reach*1.1)
        self.ax.set_ylim(-max_reach*1.1, max_reach*0.2)
        self.ax.set_zlim(-0.1, 0.1)
        
        # Plot elements
        line1, = self.ax.plot([], [], [], 'o-', linewidth=3, markersize=10, 
                             color='#E63946', label='Pendulum 1')
        line2, = self.ax.plot([], [], [], 'o-', linewidth=3, markersize=10, 
                             color='#F1FAEE', label='Pendulum 2')
        trail1, = self.ax.plot([], [], [], '-', alpha=0.3, linewidth=1, color='#E63946')
        trail2, = self.ax.plot([], [], [], '-', alpha=0.4, linewidth=1.5, color='#457B9D')
        time_text = self.ax.text2D(0.05, 0.95, '', transform=self.ax.transAxes, fontsize=12)
        
        self.ax.legend(loc='upper right')
        
        def animate_frame(frame):
            if frame < len(t):
                # First pendulum
                line1.set_data([0, x1[frame]], [0, y1[frame]])
                line1.set_3d_properties([0, 0])
                
                # Second pendulum
                line2.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
                line2.set_3d_properties([0, 0])
                
                # Trails
                trail_length = min(frame, 250)
                if trail_length > 0:
                    trail1.set_data(x1[frame-trail_length:frame+1], 
                                  y1[frame-trail_length:frame+1])
                    trail1.set_3d_properties(np.zeros(trail_length+1))
                    
                    trail2.set_data(x2[frame-trail_length:frame+1], 
                                  y2[frame-trail_length:frame+1])
                    trail2.set_3d_properties(np.zeros(trail_length+1))
                
                time_text.set_text(f'Time: {t[frame]:.2f} s')
                
            return line1, line2, trail1, trail2, time_text
        
        interval = max(1, int(1000 * (t[-1] - t[0]) / len(t)))
        self.animation = animation.FuncAnimation(
            self.fig, animate_frame, frames=len(t),
            interval=interval, blit=False, repeat=True
        )
        
        return self.animation

    def animate_oscillator(self, solution: dict, parameters: dict, system_name: str = "oscillator"):
        """Animate harmonic oscillator as mass on spring"""
        
        if not solution['success']:
            warnings.warn("Cannot animate failed simulation")
            return None
        
        t = solution['t']
        y = solution['y']
        
        x = y[0]  # Position
        v = y[1] if y.shape[0] > 1 else np.zeros_like(x)  # Velocity
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Position vs time
        ax1.set_xlim(t[0], t[-1])
        ax1.set_ylim(np.min(x)*1.2, np.max(x)*1.2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (m)')
        ax1.set_title(f'{system_name.title()} - Position vs Time')
        ax1.grid(True, alpha=0.3)
        
        line1, = ax1.plot([], [], 'b-', linewidth=2, label='Position')
        point1, = ax1.plot([], [], 'ro', markersize=8)
        ax1.legend()
        
        # Phase space
        ax2.set_xlim(np.min(x)*1.2, np.max(x)*1.2)
        ax2.set_ylim(np.min(v)*1.2, np.max(v)*1.2)
        ax2.set_xlabel('Position (m)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Phase Space')
        ax2.grid(True, alpha=0.3)
        
        line2, = ax2.plot([], [], 'g-', linewidth=1.5, alpha=0.6, label='Trajectory')
        point2, = ax2.plot([], [], 'ro', markersize=8)
        ax2.legend()
        
        def init():
            line1.set_data([], [])
            point1.set_data([], [])
            line2.set_data([], [])
            point2.set_data([], [])
            return line1, point1, line2, point2
        
        def animate_frame(frame):
            if frame < len(t):
                # Position plot
                line1.set_data(t[:frame], x[:frame])
                point1.set_data([t[frame]], [x[frame]])
                
                # Phase space
                line2.set_data(x[:frame], v[:frame])
                point2.set_data([x[frame]], [v[frame]])
            
            return line1, point1, line2, point2
        
        interval = max(1, int(1000 * (t[-1] - t[0]) / len(t)))
        self.animation = animation.FuncAnimation(
            fig, animate_frame, frames=len(t), init_func=init,
            interval=interval, blit=True, repeat=True
        )
        
        self.fig = fig
        self.ax = ax1
        
        return self.animation

    def animate(self, solution: dict, parameters: dict, system_name: str = "system"):
        """
        Generic animation dispatcher
        
        Args:
            solution: Simulation solution dictionary
            parameters: Physical parameters
            system_name: Name of the system
            
        Returns:
            Matplotlib FuncAnimation object or None
        """
        if not solution or not solution.get('success'):
            return None

        coords = solution.get('coordinates', [])
        name = (system_name or '').lower()

        try:
            # Pendulum systems
            if 'pendulum' in name or any('theta' in c for c in coords):
                return self.animate_pendulum(solution, parameters, system_name)
            
            # Oscillator systems
            elif 'oscillator' in name or 'spring' in name or (len(coords) == 1 and 'x' in coords):
                return self.animate_oscillator(solution, parameters, system_name)
            
            # Generic phase space animation for other systems
            else:
                return self._animate_phase_space(solution, system_name)
                
        except Exception as e:
            warnings.warn(f"Animation failed: {e}")
            return None
    
    def _animate_phase_space(self, solution: dict, system_name: str):
        """Generic phase space animation for arbitrary systems"""
        t = solution['t']
        y = solution['y']
        coords = solution['coordinates']
        
        if len(coords) == 0:
            return None
        
        # Use first coordinate for phase space
        q = y[0]
        q_dot = y[1] if y.shape[0] > 1 else np.zeros_like(q)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f'{system_name} - Phase Space')
        ax.set_xlabel(f'{coords[0]}')
        ax.set_ylabel(f'{coords[0]}_dot')
        ax.grid(True, alpha=0.3)
        
        line, = ax.plot([], [], 'b-', linewidth=1.5, alpha=0.6)
        point, = ax.plot([], [], 'ro', markersize=8)
        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)
        
        def init():
            ax.set_xlim(np.min(q)*1.1, np.max(q)*1.1)
            ax.set_ylim(np.min(q_dot)*1.1, np.max(q_dot)*1.1)
            line.set_data([], [])
            point.set_data([], [])
            return line, point, time_text
        
        def animate_frame(frame):
            if frame < len(t):
                line.set_data(q[:frame], q_dot[:frame])
                point.set_data([q[frame]], [q_dot[frame]])
                time_text.set_text(f'Time: {t[frame]:.2f} s')
            return line, point, time_text
        
        interval = max(1, int(1000 * (t[-1] - t[0]) / len(t)))
        self.animation = animation.FuncAnimation(
            fig, animate_frame, frames=len(t), init_func=init,
            interval=interval, blit=True, repeat=True
        )
        
        self.fig = fig
        self.ax = ax
        
        return self.animation

    def plot_energy(self, solution: dict, parameters: dict, lagrangian: sp.Expr = None):
        """
        Plot energy conservation analysis
        
        Args:
            solution: Simulation solution
            parameters: Physical parameters
            lagrangian: Optional Lagrangian for energy calculation
        """
        
        if not solution['success']:
            warnings.warn("Cannot plot energy for failed simulation")
            return
        
        t = solution['t']
        y = solution['y']
        coords = solution['coordinates']
        
        # Calculate energies
        KE = np.zeros_like(t)
        PE = np.zeros_like(t)
        
        # System-specific energy calculations
        if 'theta' in coords[0]:  # Pendulum-like
            if len(coords) == 1:  # Single pendulum
                theta = y[0]
                theta_dot = y[1]
                m = parameters.get('m', 1.0)
                l = parameters.get('l', 1.0)
                g = parameters.get('g', 9.81)
                
                KE = 0.5 * m * l**2 * theta_dot**2
                PE = m * g * l * (1 - np.cos(theta))
                
            elif len(coords) >= 2:  # Double pendulum
                theta1, theta1_dot = y[0], y[1]
                theta2, theta2_dot = y[2], y[3]
                m1 = parameters.get('m1', 1.0)
                m2 = parameters.get('m2', 1.0)
                l1 = parameters.get('l1', 1.0)
                l2 = parameters.get('l2', 1.0)
                g = parameters.get('g', 9.81)
                
                KE1 = 0.5 * m1 * l1**2 * theta1_dot**2
                KE2 = 0.5 * m2 * (l1**2 * theta1_dot**2 + l2**2 * theta2_dot**2 + 
                                  2 * l1 * l2 * theta1_dot * theta2_dot * np.cos(theta1 - theta2))
                KE = KE1 + KE2
                
                PE1 = -m1 * g * l1 * np.cos(theta1)
                PE2 = -m2 * g * (l1 * np.cos(theta1) + l2 * np.cos(theta2))
                PE = PE1 + PE2
                
        else:  # Generic oscillator
            x = y[0]
            v = y[1] if y.shape[0] > 1 else np.zeros_like(x)
            m = parameters.get('m', 1.0)
            k = parameters.get('k', 1.0)
            
            KE = 0.5 * m * v**2
            PE = 0.5 * k * x**2
        
        E_total = KE + PE
        
        # Create comprehensive energy plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Energy Analysis', fontsize=16, fontweight='bold')
        
        # Kinetic energy
        axes[0, 0].plot(t, KE, 'r-', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Energy (J)')
        axes[0, 0].set_title('Kinetic Energy')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Potential energy
        axes[0, 1].plot(t, PE, 'b-', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Energy (J)')
        axes[0, 1].set_title('Potential Energy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Total energy
        axes[1, 0].plot(t, E_total, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Energy (J)')
        axes[1, 0].set_title('Total Energy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Energy conservation error
        E_error = (E_total - E_total[0]) / E_total[0] * 100
        axes[1, 1].plot(t, E_error, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Relative Error (%)')
        axes[1, 1].set_title('Energy Conservation Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print energy conservation statistics
        print(f"\n{'='*50}")
        print("Energy Conservation Analysis")
        print(f"{'='*50}")
        print(f"Initial Total Energy: {E_total[0]:.6f} J")
        print(f"Final Total Energy:   {E_total[-1]:.6f} J")
        print(f"Energy Change:        {E_total[-1] - E_total[0]:.6e} J")
        print(f"Relative Error:       {E_error[-1]:.6f}%")
        print(f"Max Relative Error:   {np.max(np.abs(E_error)):.6f}%")
        print(f"{'='*50}\n")

    def plot_phase_space(self, solution: dict, coordinate_index: int = 0):
        """Plot phase space trajectory"""
        
        if not solution['success']:
            warnings.warn("Cannot plot phase space for failed simulation")
            return
        
        y = solution['y']
        coords = solution['coordinates']
        
        if coordinate_index >= len(coords):
            warnings.warn(f"Coordinate index {coordinate_index} out of range")
            return
        
        position = y[2 * coordinate_index]
        velocity = y[2 * coordinate_index + 1]
        
        plt.figure(figsize=(10, 10))
        plt.plot(position, velocity, 'b-', alpha=0.7, linewidth=1.5, label='Trajectory')
        plt.plot(position[0], velocity[0], 'go', markersize=10, label='Start', zorder=5)
        plt.plot(position[-1], velocity[-1], 'ro', markersize=10, label='End', zorder=5)
        
        plt.xlabel(f'{coords[coordinate_index]} (position)', fontsize=12)
        plt.ylabel(f'd{coords[coordinate_index]}/dt (velocity)', fontsize=12)
        plt.title(f'Phase Space: {coords[coordinate_index]}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

# ============================================================================
# COMPLETE PHYSICS COMPILER - The main interface
# ============================================================================

class PhysicsCompiler:
    """
    Main compiler class that orchestrates the entire DSL pipeline
    
    Usage:
        compiler = PhysicsCompiler()
        result = compiler.compile_dsl(dsl_source)
        solution = compiler.simulate((0, 10), num_points=1000)
        compiler.animate(solution)
    """
    
    def __init__(self):
        self.ast = []
        self.variables = {}
        self.definitions = {}
        self.parameters_def = {}
        self.system_name = "unnamed_system"
        self.lagrangian = None
        self.hamiltonian = None
        self.transforms = {}
        self.constraints = []
        self.initial_conditions = {}
        
        # Engines
        self.symbolic = SymbolicEngine()
        self.simulator = NumericalSimulator(self.symbolic)
        self.visualizer = MechanicsVisualizer()
        self.unit_system = UnitSystem()
        
        # Metadata
        self.compilation_time = None
        self.equations = None

    def compile_dsl(self, dsl_source: str) -> dict:
        """
        Complete compilation pipeline from DSL source to executable system
        
        Args:
            dsl_source: DSL source code string
            
        Returns:
            Dictionary with compilation results and metadata
        """
        
        start_time = time.time()
        
        try:
            # Step 1: Tokenization
            tokens = tokenize(dsl_source)
            
            # Step 2: Parsing
            parser = MechanicsParser(tokens)
            self.ast = parser.parse()
            
            if parser.errors:
                warnings.warn(f"Parser found {len(parser.errors)} errors")
            
            # Step 3: Semantic analysis
            self.analyze_semantics()
            
            # Step 4: Generate equations of motion
            equations = self.derive_equations()
            self.equations = equations
            
            # Step 5: Setup numerical simulation
            self.setup_simulation(equations)
            
            self.compilation_time = time.time() - start_time
            
            return {
                'success': True,
                'system_name': self.system_name,
                'coordinates': list(self.get_coordinates()),
                'equations': equations,
                'variables': self.variables,
                'parameters': self.simulator.parameters,
                'compilation_time': self.compilation_time,
                'ast_nodes': len(self.ast),
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'compilation_time': time.time() - start_time,
            }

    def analyze_semantics(self):
        """Extract system information from AST"""
        
        for node in self.ast:
            if isinstance(node, SystemDef):
                self.system_name = node.name
                
            elif isinstance(node, VarDef):
                self.variables[node.name] = {
                    'type': node.vartype,
                    'unit': node.unit,
                    'vector': node.vector
                }
            
            elif isinstance(node, ParameterDef):
                self.parameters_def[node.name] = {
                    'value': node.value,
                    'unit': node.unit
                }
                
            elif isinstance(node, DefineDef):
                self.definitions[node.name] = {
                    'args': node.args,
                    'body': node.body
                }
                
            elif isinstance(node, LagrangianDef):
                self.lagrangian = node.expr
                
            elif isinstance(node, HamiltonianDef):
                self.hamiltonian = node.expr
                
            elif isinstance(node, TransformDef):
                self.transforms[node.var] = {
                    'type': node.coord_type,
                    'expression': node.expr
                }
            
            elif isinstance(node, ConstraintDef):
                self.constraints.append(node.expr)
                
            elif isinstance(node, InitialCondition):
                self.initial_conditions.update(node.conditions)

    def get_coordinates(self) -> List[str]:
        """Extract generalized coordinates from variables"""
        coordinates = []
        
        for var_name, var_info in self.variables.items():
            # Identify coordinates by type or common names
            if (var_info['type'] in ['Angle', 'Position', 'Coordinate', 'Length'] or
                var_name in ['theta', 'theta1', 'theta2', 'x', 'y', 'z', 'r', 'phi', 'psi']):
                coordinates.append(var_name)
                
        return coordinates

    def derive_equations(self) -> Dict[str, sp.Expr]:
        """Derive equations of motion from Lagrangian or Hamiltonian"""
        
        if self.lagrangian is None and self.hamiltonian is None:
            raise ValueError("No Lagrangian or Hamiltonian defined in system")
        
        # Use Lagrangian formulation
        if self.lagrangian is not None:
            L_sympy = self.symbolic.ast_to_sympy(self.lagrangian)
            coordinates = self.get_coordinates()
            
            if not coordinates:
                raise ValueError("No generalized coordinates found")
            
            # Derive Euler-Lagrange equations
            eq_list = self.symbolic.derive_equations_of_motion(L_sympy, coordinates)
            
            # Solve for accelerations
            accelerations = self.symbolic.solve_for_accelerations(eq_list, coordinates)
            
            return accelerations
        
        # TODO: Implement Hamiltonian formulation
        else:
            raise NotImplementedError("Hamiltonian formulation not yet implemented")

    def setup_simulation(self, equations: Dict[str, sp.Expr]):
        """Configure numerical simulator with parameters and initial conditions"""
        
        # Extract parameters
        parameters = {}
        
        # From explicit parameter definitions
        for param_name, param_info in self.parameters_def.items():
            parameters[param_name] = param_info['value']
        
        # From variables with default values
        for var_name, var_info in self.variables.items():
            if var_info['type'] in ['Real', 'Mass', 'Length', 'Acceleration', 'Spring Constant']:
                if var_name not in parameters:
                    # Use common defaults
                    defaults = {
                        'g': 9.81,
                        'm': 1.0, 'm1': 1.0, 'm2': 1.0,
                        'l': 1.0, 'l1': 1.0, 'l2': 1.0,
                        'k': 1.0,
                    }
                    parameters[var_name] = defaults.get(var_name, 1.0)
        
        self.simulator.set_parameters(parameters)
        self.simulator.set_initial_conditions(self.initial_conditions)
        
        # Compile equations
        coordinates = self.get_coordinates()
        self.simulator.compile_equations(equations, coordinates)

    def simulate(self, t_span: Tuple[float, float] = (0, 10), 
                num_points: int = 1000, **kwargs):
        """
        Run numerical simulation
        
        Args:
            t_span: Time span (start, end)
            num_points: Number of output points
            **kwargs: Additional arguments passed to simulator
            
        Returns:
            Solution dictionary
        """
        return self.simulator.simulate(t_span, num_points, **kwargs)

    def animate(self, solution: dict, show: bool = True):
        """
        Create animation from solution
        
        Args:
            solution: Simulation solution dictionary
            show: Whether to display the animation immediately
            
        Returns:
            Animation object
        """
        parameters = self.simulator.parameters
        anim = self.visualizer.animate(solution, parameters, self.system_name)
        
        if show and anim is not None:
            plt.show()
        
        return anim

    def export_animation(self, solution: dict, filename: str, fps: int = 30, dpi: int = 100):
        """
        Export animation to file
        
        Args:
            solution: Simulation solution
            filename: Output filename (.mp4 or .gif)
            fps: Frames per second
            dpi: Resolution
            
        Returns:
            Filename on success
            
        Raises:
            RuntimeError: If animation export fails
        """
        anim = self.animate(solution, show=False)
        
        if anim is None:
            raise RuntimeError('No animation available for this solution')
        
        ok = self.visualizer.save_animation_to_file(anim, filename, fps=fps)
        
        if not ok:
            raise RuntimeError(f'Failed to save animation to {filename}')
        
        return filename

    def plot_energy(self, solution: dict):
        """Plot energy analysis"""
        self.visualizer.plot_energy(solution, self.simulator.parameters, self.lagrangian)

    def plot_phase_space(self, solution: dict, coordinate_index: int = 0):
        """Plot phase space trajectory"""
        self.visualizer.plot_phase_space(solution, coordinate_index)
    
    def print_equations(self):
        """Print derived equations of motion in readable format"""
        if self.equations is None:
            print("No equations derived yet. Run compile_dsl() first.")
            return
        
        print(f"\n{'='*70}")
        print(f"Equations of Motion: {self.system_name}")
        print(f"{'='*70}\n")
        
        for coord in self.get_coordinates():
            accel_key = f"{coord}_ddot"
            if accel_key in self.equations:
                eq = self.equations[accel_key]
                print(f"{accel_key} = {eq}\n")
        
        print(f"{'='*70}\n")
    
    def get_info(self) -> dict:
        """Get comprehensive system information"""
        return {
            'system_name': self.system_name,
            'coordinates': self.get_coordinates(),
            'variables': self.variables,
            'parameters': self.simulator.parameters,
            'initial_conditions': self.initial_conditions,
            'has_lagrangian': self.lagrangian is not None,
            'has_hamiltonian': self.hamiltonian is not None,
            'num_constraints': len(self.constraints),
            'compilation_time': self.compilation_time,
        }


# ============================================================================
# EXAMPLE SYSTEMS AND UTILITIES
# ============================================================================

def example_simple_pendulum() -> str:
    """Example DSL code for simple pendulum"""
    return r"""
\system{simple_pendulum}

\defvar{theta}{Angle}{rad}
\defvar{m}{Mass}{kg}
\defvar{l}{Length}{m}
\defvar{g}{Acceleration}{m/s^2}

\parameter{m}{1.0}{kg}
\parameter{l}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

\lagrangian{\frac{1}{2} m l^2 \dot{theta}^2 - m g l (1 - \cos{\theta})}

\initial{theta=0.5, theta_dot=0.0}

\solve{RK45}
\animate{pendulum}
"""

def example_double_pendulum() -> str:
    """Example DSL code for double pendulum"""
    return r"""
\system{double_pendulum}

\defvar{theta1}{Angle}{rad}
\defvar{theta2}{Angle}{rad}
\defvar{m1}{Mass}{kg}
\defvar{m2}{Mass}{kg}
\defvar{l1}{Length}{m}
\defvar{l2}{Length}{m}
\defvar{g}{Acceleration}{m/s^2}

\parameter{m1}{1.0}{kg}
\parameter{m2}{1.0}{kg}
\parameter{l1}{1.0}{m}
\parameter{l2}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

\lagrangian{
    \frac{1}{2} (m1 + m2) l1^2 \dot{theta1}^2 
    + \frac{1}{2} m2 l2^2 \dot{theta2}^2 
    + m2 l1 l2 \dot{theta1} \dot{theta2} \cos{theta1 - theta2}
    + (m1 + m2) g l1 \cos{theta1}
    + m2 g l2 \cos{theta2}
}

\initial{theta1=1.57, theta1_dot=0.0, theta2=1.57, theta2_dot=0.0}

\solve{RK45}
\animate{double_pendulum}
"""

def example_harmonic_oscillator() -> str:
    """Example DSL code for harmonic oscillator"""
    return r"""
\system{harmonic_oscillator}

\defvar{x}{Position}{m}
\defvar{m}{Mass}{kg}
\defvar{k}{Spring Constant}{N/m}

\parameter{m}{1.0}{kg}
\parameter{k}{10.0}{N/m}

\lagrangian{\frac{1}{2} m \dot{x}^2 - \frac{1}{2} k x^2}

\initial{x=1.0, x_dot=0.0}

\solve{RK45}
\animate{oscillator}
"""

def run_example(example_name: str = "simple_pendulum", 
                t_span: Tuple[float, float] = (0, 10),
                show_animation: bool = True,
                show_energy: bool = True,
                show_phase: bool = True) -> dict:
    """
    Run a built-in example system
    
    Args:
        example_name: Name of example ('simple_pendulum', 'double_pendulum', 'harmonic_oscillator')
        t_span: Time span for simulation
        show_animation: Whether to show animation
        show_energy: Whether to show energy plot
        show_phase: Whether to show phase space plot
        
    Returns:
        Dictionary with compiler and solution
    """
    
    examples = {
        'simple_pendulum': example_simple_pendulum(),
        'double_pendulum': example_double_pendulum(),
        'harmonic_oscillator': example_harmonic_oscillator(),
    }
    
    if example_name not in examples:
        raise ValueError(f"Unknown example: {example_name}. Choose from {list(examples.keys())}")
    
    dsl_code = examples[example_name]
    
    # Compile
    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(dsl_code)
    
    if not result['success']:
        print(f"Compilation failed: {result.get('error', 'Unknown error')}")
        return {'compiler': compiler, 'solution': None}
    
    print(f"\n{'='*70}")
    print(f"Successfully compiled: {result['system_name']}")
    print(f"Coordinates: {result['coordinates']}")
    print(f"Compilation time: {result['compilation_time']:.4f} seconds")
    print(f"{'='*70}\n")
    
    # Print equations
    compiler.print_equations()
    
    # Simulate
    print("Running simulation...")
    solution = compiler.simulate(t_span, num_points=1000)
    
    if not solution['success']:
        print(f"Simulation failed: {solution.get('error', 'Unknown error')}")
        return {'compiler': compiler, 'solution': solution}
    
    print(f"Simulation completed: {solution['nfev']} function evaluations")
    
    # Visualize
    if show_animation:
        print("\nCreating animation...")
        compiler.animate(solution, show=True)
    
    if show_energy:
        print("\nPlotting energy analysis...")
        compiler.plot_energy(solution)
    
    if show_phase:
        print("\nPlotting phase space...")
        compiler.plot_phase_space(solution, coordinate_index=0)
    
    return {
        'compiler': compiler,
        'solution': solution,
        'result': result
    }


# ============================================================================
# VALIDATION AND TESTING UTILITIES
# ============================================================================

class SystemValidator:
    """Validate DSL systems against known analytical solutions"""
    
    @staticmethod
    def validate_simple_harmonic_oscillator(compiler: PhysicsCompiler, 
                                           solution: dict,
                                           tolerance: float = 0.01) -> bool:
        """
        Validate harmonic oscillator against analytical solution
        x(t) = A cos(ωt + φ)
        """
        if not solution['success']:
            return False
        
        t = solution['t']
        x = solution['y'][0]
        v = solution['y'][1]
        
        # Extract parameters
        m = compiler.simulator.parameters.get('m', 1.0)
        k = compiler.simulator.parameters.get('k', 1.0)
        
        # Calculate natural frequency
        omega = np.sqrt(k / m)
        
        # Extract amplitude and phase from initial conditions
        x0 = x[0]
        v0 = v[0]
        A = np.sqrt(x0**2 + (v0/omega)**2)
        phi = np.arctan2(-v0/omega, x0)
        
        # Analytical solution
        x_analytical = A * np.cos(omega * t + phi)
        
        # Calculate error
        error = np.max(np.abs(x - x_analytical)) / A
        
        print(f"\nHarmonic Oscillator Validation:")
        print(f"  Max relative error: {error:.6f}")
        print(f"  Tolerance: {tolerance}")
        print(f"  Passed: {error < tolerance}")
        
        return error < tolerance
    
    @staticmethod
    def validate_energy_conservation(compiler: PhysicsCompiler,
                                    solution: dict,
                                    tolerance: float = 0.01) -> bool:
        """
        Validate energy conservation for conservative systems
        """
        if not solution['success']:
            return False
        
        t = solution['t']
        y = solution['y']
        coords = solution['coordinates']
        params = compiler.simulator.parameters
        
        # Calculate energy based on system type
        if 'theta' in coords[0]:  # Pendulum
            theta = y[0]
            theta_dot = y[1]
            m = params.get('m', 1.0)
            l = params.get('l', 1.0)
            g = params.get('g', 9.81)
            
            KE = 0.5 * m * l**2 * theta_dot**2
            PE = m * g * l * (1 - np.cos(theta))
            
        else:  # Oscillator
            x = y[0]
            v = y[1]
            m = params.get('m', 1.0)
            k = params.get('k', 1.0)
            
            KE = 0.5 * m * v**2
            PE = 0.5 * k * x**2
        
        E_total = KE + PE
        E_error = np.abs((E_total - E_total[0]) / E_total[0])
        max_error = np.max(E_error)
        
        print(f"\nEnergy Conservation Validation:")
        print(f"  Max relative energy error: {max_error:.6f}")
        print(f"  Tolerance: {tolerance}")
        print(f"  Passed: {max_error < tolerance}")
        
        return max_error < tolerance


# ============================================================================
# COMMAND LINE INTERFACE (Optional)
# ============================================================================

def main():
    """Command line interface for MechanicsDSL"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='MechanicsDSL - Domain-Specific Language for Classical Mechanics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run built-in example
  python mechanics_dsl.py --example simple_pendulum
  
  # Compile and simulate custom DSL file
  python mechanics_dsl.py --file my_system.dsl --time 20 --export animation.mp4
  
  # Run with energy and phase space analysis
  python mechanics_dsl.py --example double_pendulum --energy --phase
        """
    )
    
    parser.add_argument('--example', type=str, choices=['simple_pendulum', 'double_pendulum', 'harmonic_oscillator'],
                       help='Run a built-in example system')
    parser.add_argument('--file', type=str, help='DSL source file to compile')
    parser.add_argument('--time', type=float, default=10.0, help='Simulation time (default: 10.0)')
    parser.add_argument('--points', type=int, default=1000, help='Number of time points (default: 1000)')
    parser.add_argument('--export', type=str, help='Export animation to file (.mp4 or .gif)')
    parser.add_argument('--energy', action='store_true', help='Show energy analysis')
    parser.add_argument('--phase', action='store_true', help='Show phase space plot')
    parser.add_argument('--validate', action='store_true', help='Run validation tests')
    parser.add_argument('--no-animation', action='store_true', help='Skip animation display')
    
    args = parser.parse_args()
    
    # Determine what to run
    if args.example:
        # Run built-in example
        results = run_example(
            args.example,
            t_span=(0, args.time),
            show_animation=not args.no_animation,
            show_energy=args.energy,
            show_phase=args.phase
        )
        
        compiler = results['compiler']
        solution = results['solution']
        
    elif args.file:
        # Load and compile custom file
        try:
            with open(args.file, 'r') as f:
                dsl_code = f.read()
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found")
            return 1
        
        compiler = PhysicsCompiler()
        result = compiler.compile_dsl(dsl_code)
        
        if not result['success']:
            print(f"Compilation failed: {result.get('error', 'Unknown error')}")
            return 1
        
        print(f"Successfully compiled: {result['system_name']}")
        compiler.print_equations()
        
        solution = compiler.simulate((0, args.time), num_points=args.points)
        
        if not solution['success']:
            print(f"Simulation failed: {solution.get('error', 'Unknown error')}")
            return 1
        
        if not args.no_animation:
            compiler.animate(solution, show=True)
        
        if args.energy:
            compiler.plot_energy(solution)
        
        if args.phase:
            compiler.plot_phase_space(solution)
    
    else:
        parser.print_help()
        return 0
    
    # Export animation if requested
    if args.export and solution and solution['success']:
        print(f"\nExporting animation to {args.export}...")
        try:
            compiler.export_animation(solution, args.export)
            print(f"Animation saved successfully!")
        except Exception as e:
            print(f"Export failed: {e}")
    
    # Run validation if requested
    if args.validate and solution and solution['success']:
        print("\n" + "="*70)
        print("Running validation tests...")
        print("="*70)
        
        validator = SystemValidator()
        
        if 'oscillator' in compiler.system_name.lower():
            validator.validate_simple_harmonic_oscillator(compiler, solution)
        
        validator.validate_energy_conservation(compiler, solution)
    
    return 0


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'PhysicsCompiler',
    'SymbolicEngine',
    'NumericalSimulator',
    'MechanicsVisualizer',
    'SystemValidator',
    'tokenize',
    'MechanicsParser',
    'example_simple_pendulum',
    'example_double_pendulum',
    'example_harmonic_oscillator',
    'run_example',
]


if __name__ == '__main__':
    import sys
    
    # If no arguments, run interactive demo
    if len(sys.argv) == 1:
        print("""
╔════════════════════════════════════════════════════════════════════╗
║                         MechanicsDSL v0.1.0                        ║
║        A Domain-Specific Language for Classical Mechanics          ║
╚════════════════════════════════════════════════════════════════════╝

Running interactive demo with simple pendulum...
        """)
        
        results = run_example('simple_pendulum', t_span=(0, 10))
        
        print("\n" + "="*70)
        print("Demo completed! Try running with --help to see more options.")
        print("="*70)
    else:
        sys.exit(main())
errors.append(error_msg)
                raise SyntaxError(error_msg)
            else:
                error_msg = f"Expected {expected_type} but reached end of input"
                self.errors.append(error_msg)
                raise SyntaxError(error_msg)
        return token

    def parse(self) -> List[ASTNode]:
        """Parse the complete DSL with comprehensive error recovery"""
        nodes = []
        while self.pos < len(self.tokens):
            try:
                node = self.parse_statement()
                if node:
                    nodes.append(node)
            except SyntaxError as e:
                self.errors.append(str(e))
                # Try to recover by skipping to next statement
                while self.pos < len(self.tokens):
                    token = self.peek()
                    if token and token.type in ["SYSTEM", "DEFVAR", "DEFINE", "LAGRANGIAN"]:
                        break
                    self.pos += 1
        
        if self.errors:
            warnings.warn(f"Parser encountered {len(self.errors)} errors")
            
        return nodes

    def parse_statement(self) -> Optional[ASTNode]:
        """Parse a top-level statement"""
        token = self.peek()
        if not token:
            return None

        handlers = {
            "SYSTEM": self.parse_system,
            "DEFVAR": self.parse_defvar,
            "PARAMETER": self.parse_parameter,
            "DEFINE": self.parse_define,
            "LAGRANGIAN": self.parse_lagrangian,
            "HAMILTONIAN": self.parse_hamiltonian,
            "TRANSFORM": self.parse_transform,
            "CONSTRAINT": self.parse_constraint,
            "INITIAL": self.parse_initial,
            "SOLVE": self.parse_solve,
            "ANIMATE": self.parse_animate,
        }
        
        handler = handlers.get(token.type)
        if handler:
            return handler()
        else:
            # Skip unknown tokens
            self.pos += 1
            return None

    def parse_system(self) -> SystemDef:
        """Parse \\system{name}"""
        self.expect("SYSTEM")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        self.current_system = name
        return SystemDef(name)

    def parse_defvar(self) -> VarDef:
        """Parse \\defvar{name}{type}{unit}"""
        self.expect("DEFVAR")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        self.expect("LBRACE")
        
        # Collect vartype (may be multiple tokens)
        vartype_parts = []
        while True:
            tok = self.peek()
            if not tok or tok.type == 'RBRACE':
                break
            self.pos += 1
            vartype_parts.append(tok.value)
        vartype = ' '.join(vartype_parts).strip()
        self.expect("RBRACE")
        
        self.expect("LBRACE")
        unit_expr = self.parse_expression()
        unit = self.expression_to_string(unit_expr)
        self.expect("RBRACE")
        
        # Check if it's a vector type
        is_vector = vartype in ["Vector", "Vector3", "Position", "Velocity", "Force", "Momentum"]
        
        return VarDef(name, vartype, unit, is_vector)
    
    def parse_parameter(self) -> ParameterDef:
        """Parse \\parameter{name}{value}{unit}"""
        self.expect("PARAMETER")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        self.expect("LBRACE")
        value = float(self.expect("NUMBER").value)
        self.expect("RBRACE")
        self.expect("LBRACE")
        unit = self.expect("IDENT").value  # Simplified
        self.expect("RBRACE")
        return ParameterDef(name, value, unit)

    def parse_define(self) -> DefineDef:
        """Parse \\define{\\op{name}(args) = expression}"""
        self.expect("DEFINE")
        self.expect("LBRACE")
        
        # Expect \\op{name} or similar
        self.expect("COMMAND")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        
        # Parse arguments
        self.expect("LPAREN")
        args = []
        if self.peek() and self.peek().type == "IDENT":
            args.append(self.expect("IDENT").value)
            while self.match("COMMA"):
                args.append(self.expect("IDENT").value)
        self.expect("RPAREN")
        
        self.expect("EQUALS")
        body = self.parse_expression()
        self.expect("RBRACE")
        
        return DefineDef(name, args, body)

    def parse_lagrangian(self) -> LagrangianDef:
        """Parse \\lagrangian{expression}"""
        self.expect("LAGRANGIAN")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return LagrangianDef(expr)

    def parse_hamiltonian(self) -> HamiltonianDef:
        """Parse \\hamiltonian{expression}"""
        self.expect("HAMILTONIAN")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return HamiltonianDef(expr)

    def parse_transform(self) -> TransformDef:
        """Parse \\transform{type}{var = expr}"""
        self.expect("TRANSFORM")
        self.expect("LBRACE")
        coord_type = self.expect("IDENT").value
        self.expect("RBRACE")
        self.expect("LBRACE")
        var = self.expect("IDENT").value
        self.expect("EQUALS")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return TransformDef(coord_type, var, expr)
    
    def parse_constraint(self) -> ConstraintDef:
        """Parse \\constraint{expression}"""
        self.expect("CONSTRAINT")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return ConstraintDef(expr)

    def parse_initial(self) -> InitialCondition:
        """Parse \\initial{var1=val1, var2=val2, ...}"""
        self.expect("INITIAL")
        self.expect("LBRACE")
        
        conditions = {}
        var = self.expect("IDENT").value
        self.expect("EQUALS")
        val = float(self.expect("NUMBER").value)
        conditions[var] = val
        
        while self.match("COMMA"):
            var = self.expect("IDENT").value
            self.expect("EQUALS")
            val = float(self.expect("NUMBER").value)
            conditions[var] = val
            
        self.expect("RBRACE")
        return InitialCondition(conditions)

    def parse_solve(self) -> SolveDef:
        """Parse \\solve{method}"""
        self.expect("SOLVE")
        self.expect("LBRACE")
        method = self.expect("IDENT").value
        self.expect("RBRACE")
        return SolveDef(method)

    def parse_animate(self) -> AnimateDef:
        """Parse \\animate{target}"""
        self.expect("ANIMATE")
        self.expect("LBRACE")
        target = self.expect("IDENT").value
        self.expect("RBRACE")
        return AnimateDef(target)

    # Expression parsing with full precedence
    def parse_expression(self) -> Expression:
        """Parse expressions with full operator precedence"""
        return self.parse_additive()

    def parse_additive(self) -> Expression:
        """Addition and subtraction"""
        left = self.parse_multiplicative()
        
        while True:
            if self.match("PLUS"):
                right = self.parse_multiplicative()
                left = BinaryOpExpr(left, "+", right)
            elif self.match("MINUS"):
                right = self.parse_multiplicative()
                left = BinaryOpExpr(left, "-", right)
            else:
                break
                
        return left

    def parse_multiplicative(self) -> Expression:
        """Multiplication, division, and implicit multiplication"""
        left = self.parse_power()
        
        while True:
            if self.match("MULTIPLY"):
                right = self.parse_power()
                left = BinaryOpExpr(left, "*", right)
            elif self.match("DIVIDE"):
                right = self.parse_power()
                left = BinaryOpExpr(left, "/", right)
            elif self.match("VECTOR_DOT"):
                # Dot product
                right = self.parse_power()
                left = VectorOpExpr("dot", left, right)
            elif self.match("VECTOR_CROSS"):
                # Cross product
                right = self.parse_power()
                left = VectorOpExpr("cross", left, right)
            else:
                # Check for implicit multiplication
                next_token = self.peek()
                if (next_token and 
                    next_token.type in ["IDENT", "NUMBER", "LPAREN", "GREEK_LETTER", "COMMAND"] and
                    not self.at_end_of_expression()):
                    right = self.parse_power()
                    left = BinaryOpExpr(left, "*", right)
                else:
                    break
                    
        return left

    def parse_power(self) -> Expression:
        """Exponentiation (right associative)"""
        left = self.parse_unary()
        
        if self.match("POWER"):
            right = self.parse_power()  # Right associative
            return BinaryOpExpr(left, "^", right)
            
        return left

    def parse_unary(self) -> Expression:
        """Unary operators"""
        if self.match("MINUS"):
            operand = self.parse_unary()
            return UnaryOpExpr("-", operand)
        elif self.match("PLUS"):
            return self.parse_unary()
        
        return self.parse_postfix()

    def parse_postfix(self) -> Expression:
        """Function calls, derivatives, etc."""
        expr = self.parse_primary()
        
        while True:
            if self.match("LPAREN"):
                # Function call
                args = []
                if not self.peek() or self.peek().type != "RPAREN":
                    args.append(self.parse_expression())
                    while self.match("COMMA"):
                        args.append(self.parse_expression())
                self.