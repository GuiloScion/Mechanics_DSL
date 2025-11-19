"""
MechanicsDSL: A Domain-Specific Language for Classical Mechanics

A comprehensive framework for symbolic and numerical analysis of classical 
mechanical systems using LaTeX-inspired notation.

Author: Noah Parsons
Version: 0.2.0 - Major improvements and bug fixes
"""

import re
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from typing import List, Dict, Optional, Tuple, Any, Union, Callable, Literal
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import warnings
from collections import defaultdict

__version__ = "0.2.0"
__author__ = "Noah Parsons"

# ============================================================================
# TOKEN SYSTEM - Enhanced with better coverage and ordering
# ============================================================================

TOKEN_TYPES = [
    # Physics specific commands (order matters!)
    ("DOT_NOTATION", r"\\ddot|\\dot"),  # Combined and ordered before DOT
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
    ("PARAMETER", r"\\parameter"),
    
    # Vector operations
    ("VEC", r"\\vec"),
    ("HAT", r"\\hat"),
    ("MAGNITUDE", r"\\mag|\\norm"),
    
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
    ("FRAC", r"\\frac"),
    
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
    
    # Mathematical operators (DOT moved to top as DOT_NOTATION)
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
@dataclass
class NumberExpr(Expression):
    value: float
    def __repr__(self):
        return f"Num({self.value})"

@dataclass
class IdentExpr(Expression):
    name: str
    def __repr__(self):
        return f"Id({self.name})"

@dataclass
class GreekLetterExpr(Expression):
    letter: str
    def __repr__(self):
        return f"Greek({self.letter})"

@dataclass
class DerivativeVarExpr(Expression):
    """Represents \dot{x} or \ddot{x} notation"""
    var: str
    order: int = 1
    def __repr__(self):
        return f"DerivativeVar({self.var}, order={self.order})"

# Binary operations with type safety
@dataclass
class BinaryOpExpr(Expression):
    left: Expression
    operator: Literal["+", "-", "*", "/", "^"]
    right: Expression
    def __repr__(self):
        return f"BinOp({self.left} {self.operator} {self.right})"

@dataclass
class UnaryOpExpr(Expression):
    operator: Literal["+", "-"]
    operand: Expression
    def __repr__(self):
        return f"UnaryOp({self.operator}{self.operand})"

# Vector expressions
@dataclass
class VectorExpr(Expression):
    components: List[Expression]
    def __repr__(self):
        return f"Vector({self.components})"

@dataclass
class VectorOpExpr(Expression):
    operation: str
    left: Expression
    right: Optional[Expression] = None
    def __repr__(self):
        if self.right:
            return f"VectorOp({self.operation}, {self.left}, {self.right})"
        return f"VectorOp({self.operation}, {self.left})"

# Calculus expressions
@dataclass
class DerivativeExpr(Expression):
    expr: Expression
    var: str
    order: int = 1
    partial: bool = False
    def __repr__(self):
        type_str = "Partial" if self.partial else "Total"
        return f"{type_str}Deriv({self.expr}, {self.var}, order={self.order})"

@dataclass
class IntegralExpr(Expression):
    expr: Expression
    var: str
    lower: Optional[Expression] = None
    upper: Optional[Expression] = None
    line_integral: bool = False
    def __repr__(self):
        return f"Integral({self.expr}, {self.var}, {self.lower}, {self.upper})"

# Function calls
@dataclass
class FunctionCallExpr(Expression):
    name: str
    args: List[Expression]
    def __repr__(self):
        return f"Call({self.name}, {self.args})"

@dataclass
class FractionExpr(Expression):
    numerator: Expression
    denominator: Expression
    def __repr__(self):
        return f"Frac({self.numerator}/{self.denominator})"

# Physics-specific AST nodes
@dataclass
class SystemDef(ASTNode):
    name: str
    def __repr__(self):
        return f"System({self.name})"

@dataclass
class VarDef(ASTNode):
    name: str
    vartype: str
    unit: str
    vector: bool = False
    def __repr__(self):
        vec_str = " [Vector]" if self.vector else ""
        return f"VarDef({self.name}: {self.vartype}[{self.unit}]{vec_str})"

@dataclass
class ParameterDef(ASTNode):
    name: str
    value: float
    unit: str
    def __repr__(self):
        return f"Parameter({self.name} = {self.value} [{self.unit}])"

@dataclass
class DefineDef(ASTNode):
    name: str
    args: List[str]
    body: Expression
    def __repr__(self):
        return f"Define({self.name}({', '.join(self.args)}) = {self.body})"

@dataclass
class LagrangianDef(ASTNode):
    expr: Expression
    def __repr__(self):
        return f"Lagrangian({self.expr})"

@dataclass
class HamiltonianDef(ASTNode):
    expr: Expression
    def __repr__(self):
        return f"Hamiltonian({self.expr})"

@dataclass
class TransformDef(ASTNode):
    coord_type: str
    var: str
    expr: Expression
    def __repr__(self):
        return f"Transform({self.coord_type}: {self.var} = {self.expr})"

@dataclass
class ConstraintDef(ASTNode):
    expr: Expression
    constraint_type: str = "holonomic"
    def __repr__(self):
        return f"Constraint({self.expr}, type={self.constraint_type})"

@dataclass
class InitialCondition(ASTNode):
    conditions: Dict[str, float]
    def __repr__(self):
        return f"Initial({self.conditions})"

@dataclass
class SolveDef(ASTNode):
    method: str
    options: Dict[str, Any] = field(default_factory=dict)
    def __repr__(self):
        return f"Solve({self.method}, {self.options})"

@dataclass
class AnimateDef(ASTNode):
    target: str
    options: Dict[str, Any] = field(default_factory=dict)
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
    "N": Unit({"mass": 1, "length": 1, "time": -2}),
    "J": Unit({"mass": 1, "length": 2, "time": -2}),
    "W": Unit({"mass": 1, "length": 2, "time": -3}),
    "Pa": Unit({"mass": 1, "length": -1, "time": -2}),
    "Hz": Unit({"time": -1}),
    "C": Unit({"current": 1, "time": 1}),
    "V": Unit({"mass": 1, "length": 2, "time": -3, "current": -1}),
    "F": Unit({"mass": -1, "length": -2, "time": 4, "current": 2}),
    "Wb": Unit({"mass": 1, "length": 2, "time": -2, "current": -1}),
    "T": Unit({"mass": 1, "time": -2, "current": -1}),
    
    # Angle units
    "rad": Unit({"angle": 1}),
    "deg": Unit({"angle": 1}, scale=np.pi/180),
}

class UnitSystem:
    """Manages unit operations and conversions"""
    
    def __init__(self):
        self.units = BASE_UNITS.copy()
        
    def parse_unit(self, unit_str: str) -> Unit:
        """Parse unit string like 'kg*m/s^2' into Unit object"""
        if unit_str in self.units:
            return self.units[unit_str]
        
        try:
            if '*' in unit_str or '/' in unit_str or '^' in unit_str:
                namespace = {k: v for k, v in self.units.items()}
                return eval(unit_str, {"__builtins__": {}}, namespace)
            return Unit({})
        except:
            warnings.warn(f"Could not parse unit: {unit_str}")
            return Unit({})
    
    def check_compatibility(self, unit1: str, unit2: str) -> bool:
        """Check if two units are compatible"""
        u1 = self.parse_unit(unit1)
        u2 = self.parse_unit(unit2)
        return u1.is_compatible(u2)

# ============================================================================
# ENHANCED PARSER ENGINE - FIXED
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
                self.errors.append(error_msg)
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
        unit = self.expect("IDENT").value
        self.expect("RBRACE")
        return ParameterDef(name, value, unit)

    def parse_define(self) -> DefineDef:
        """Parse \\define{\\op{name}(args) = expression}"""
        self.expect("DEFINE")
        self.expect("LBRACE")
        
        self.expect("COMMAND")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        
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
        """Multiplication, division, and explicit products only"""
        left = self.parse_power()
        
        while True:
            if self.match("MULTIPLY"):
                right = self.parse_power()
                left = BinaryOpExpr(left, "*", right)
            elif self.match("DIVIDE"):
                right = self.parse_power()
                left = BinaryOpExpr(left, "/", right)
            elif self.match("VECTOR_DOT"):
                right = self.parse_power()
                left = VectorOpExpr("dot", left, right)
            elif self.match("VECTOR_CROSS"):
                right = self.parse_power()
                left = VectorOpExpr("cross", left, right)
            else:
                # Check for implicit multiplication (more conservative)
                next_token = self.peek()
                if (next_token and 
                    next_token.type in ["LPAREN"] and
                    not self.at_end_of_expression()):
                    # Only allow implicit mult before parentheses
                    right = self.parse_power()
                    left = BinaryOpExpr(left, "*", right)
                else:
                    break
                    
        return left

    def parse_power(self) -> Expression:
        """Exponentiation (right associative)"""
        left = self.parse_unary()
        
        if self.match("POWER"):
            right = self.parse_power()
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
        """Function calls, subscripts, etc. - FIXED"""
        expr = self.parse_primary()
        
        while True:
            if self.match("LPAREN"):
                # Function call
                args = []
                if self.peek() and self.peek().type != "RPAREN":
                    args.append(self.parse_expression())
                    while self.match("COMMA"):
                        args.append(self.parse_expression())
                self.expect("RPAREN")
                
                if isinstance(expr, IdentExpr):
                    expr = FunctionCallExpr(expr.name, args)
                elif isinstance(expr, GreekLetterExpr):
                    expr = FunctionCallExpr(expr.letter, args)
                else:
                    raise SyntaxError("Invalid function call syntax")
            else:
                break
                
        return expr

    def parse_primary(self) -> Expression:
        """Primary expressions: literals, identifiers, parentheses, vectors, commands"""

        # Numbers
        if self.match("NUMBER"):
            return NumberExpr(float(self.tokens[self.pos - 1].value))

        # Time derivatives: \dot{x} and \ddot{x} - FIXED
        token = self.peek()
        if token and token.type == "DOT_NOTATION":
            self.pos += 1
            order = 2 if token.value == r"\ddot" else 1
            self.expect("LBRACE")
            var = self.expect("IDENT").value
            self.expect("RBRACE")
            return DerivativeVarExpr(var, order)

        # Identifiers
        if self.match("IDENT"):
            return IdentExpr(self.tokens[self.pos - 1].value)

        # Greek letters
        if self.match("GREEK_LETTER"):
            letter = self.tokens[self.pos - 1].value[1:]
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
            func_name = cmd[1:]
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
    Enhanced symbolic mathematics engine with caching and better error handling
    """
    
    def __init__(self):
        self.sp = sp
        self.symbol_map = {}
        self.function_map = {}
        self.time_symbol = sp.Symbol('t', real=True)
        self.assumptions = {}

    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a SymPy symbol with assumptions (cached)"""
        if name not in self.symbol_map:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self.symbol_map[name] = sp.Symbol(name, **default_assumptions)
            self.assumptions[name] = default_assumptions
        return self.symbol_map[name]

    def get_function(self, name: str) -> sp.Function:
        """Get or create a SymPy function (cached)"""
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
                if expr.var == "t":
                    return sp.diff(inner, self.time_symbol, expr.order)
                else:
                    return sp.diff(inner, var, expr.order)
                    
        elif isinstance(expr, FunctionCallExpr):
            args = [self.ast_to_sympy(arg) for arg in expr.args]
            
            builtin_funcs = {
                "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                "exp": sp.exp, "log": sp.log, "ln": sp.log,
                "sqrt": sp.sqrt, "sinh": sp.sinh, "cosh": sp.cosh,
                "tanh": sp.tanh, "arcsin": sp.asin, "arccos": sp.acos,
                "arctan": sp.atan, "abs": sp.Abs,
            }
            
            if expr.name in builtin_funcs:
                return builtin_funcs[expr.name](*args)
            else:
                func = self.get_function(expr.name)
                return func(*args)
                
        elif isinstance(expr, VectorExpr):
            return sp.Matrix([self.ast_to_sympy(comp) for comp in expr.components])
            
        elif isinstance(expr, VectorOpExpr):
            if expr.operation == "grad":
                if expr.left:
                    inner = self.ast_to_sympy(expr.left)
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
        
        Args:
            lagrangian: Lagrangian expression
            coordinates: List of generalized coordinates
            
        Returns:
            List of equations of motion
        """
        equations = []
        
        for q in coordinates:
            q_sym = self.get_symbol(q)
            q_dot_sym = self.get_symbol(f"{q}_dot")
            q_ddot_sym = self.get_symbol(f"{q}_ddot")

            q_func = sp.Function(q)(self.time_symbol)

            L_with_funcs = lagrangian.subs(q_sym, q_func)
            L_with_funcs = L_with_funcs.subs(q_dot_sym, sp.diff(q_func, self.time_symbol))

            dL_dq_dot = sp.diff(L_with_funcs, sp.diff(q_func, self.time_symbol))
            d_dt_dL_dq_dot = sp.diff(dL_dq_dot, self.time_symbol)
            dL_dq = sp.diff(L_with_funcs, q_func)

            equation = d_dt_dL_dq_dot - dL_dq

            equation = equation.subs(q_func, q_sym)
            equation = equation.subs(sp.diff(q_func, self.time_symbol), q_dot_sym)
            equation = equation.subs(sp.diff(q_func, self.time_symbol, 2), q_ddot_sym)

            equation = sp.simplify(equation)
            equations.append(equation)
            
        return equations

    def derive_hamiltonian_equations(self, hamiltonian: sp.Expr, 
                                    coordinates: List[str]) -> Tuple[List[sp.Expr], List[sp.Expr]]:
        """
        Derive Hamilton's equations from Hamiltonian
        
        Hamilton's equations:
        dq/dt = ∂H/∂p
        dp/dt = -∂H/∂q
        
        Args:
            hamiltonian: Hamiltonian expression
            coordinates: List of generalized coordinates
            
        Returns:
            Tuple of (q_dot equations, p_dot equations)
        """
        q_dot_equations = []
        p_dot_equations = []
        
        for q in coordinates:
            q_sym = self.get_symbol(q)
            p_sym = self.get_symbol(f"p_{q}")
            
            # dq/dt = ∂H/∂p
            q_dot = sp.diff(hamiltonian, p_sym)
            q_dot_equations.append(sp.simplify(q_dot))
            
            # dp/dt = -∂H/∂q
            p_dot = -sp.diff(hamiltonian, q_sym)
            p_dot_equations.append(sp.simplify(p_dot))
            
        return q_dot_equations, p_dot_equations

    def lagrangian_to_hamiltonian(self, lagrangian: sp.Expr, 
                                 coordinates: List[str]) -> sp.Expr:
        """
        Convert Lagrangian to Hamiltonian via Legendre transform
        
        H = Σ(p_i * q̇_i) - L
        where p_i = ∂L/∂q̇_i
        
        Args:
            lagrangian: Lagrangian expression
            coordinates: List of generalized coordinates
            
        Returns:
            Hamiltonian expression
        """
        hamiltonian = sp.S.Zero
        
        for q in coordinates:
            q_dot_sym = self.get_symbol(f"{q}_dot")
            
            # Calculate conjugate momentum p = ∂L/∂q̇
            p_sym = self.get_symbol(f"p_{q}")
            momentum_def = sp.diff(lagrangian, q_dot_sym)
            
            # Solve for q̇ in terms of p (if possible)
            try:
                q_dot_solution = sp.solve(momentum_def - p_sym, q_dot_sym)
                if q_dot_solution:
                    q_dot_expr = q_dot_solution[0]
                    # H += p * q̇
                    hamiltonian += p_sym * q_dot_expr
            except:
                # If we can't solve analytically, use implicit form
                hamiltonian += p_sym * q_dot_sym
        
        # H = Σ(p_i * q̇_i) - L
        hamiltonian = hamiltonian - lagrangian
        
        # Substitute momentum definitions
        for q in coordinates:
            q_dot_sym = self.get_symbol(f"{q}_dot")
            p_sym = self.get_symbol(f"p_{q}")
            momentum_def = sp.diff(lagrangian, q_dot_sym)
            
            try:
                q_dot_solution = sp.solve(momentum_def - p_sym, q_dot_sym)
                if q_dot_solution:
                    hamiltonian = hamiltonian.subs(q_dot_sym, q_dot_solution[0])
            except:
                pass
        
        return sp.simplify(hamiltonian)

    def solve_for_accelerations(self, equations: List[sp.Expr], 
                               coordinates: List[str]) -> Dict[str, sp.Expr]:
        """
        Solve equations of motion for accelerations
        
        Args:
            equations: List of equations of motion
            coordinates: List of generalized coordinates
            
        Returns:
            Dictionary mapping acceleration symbols to expressions
        """
        accelerations = {}
        accel_symbols = [self.get_symbol(f"{q}_ddot") for q in coordinates]
        
        try:
            solutions = sp.solve(equations, accel_symbols, dict=True)
            
            if solutions:
                sol = solutions[0] if isinstance(solutions, list) else solutions
                for q in coordinates:
                    accel_sym = self.get_symbol(f"{q}_ddot")
                    if accel_sym in sol:
                        accelerations[f"{q}_ddot"] = sp.simplify(sol[accel_sym])
            else:
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
            for i, q in enumerate(coordinates):
                accelerations[f"{q}_ddot"] = equations[i]
                
        return accelerations

# ============================================================================
# ENHANCED NUMERICAL SIMULATION ENGINE
# ============================================================================

class NumericalSimulator:
    """
    Enhanced numerical simulator with better stability and diagnostics
    """
    
    def __init__(self, symbolic_engine: SymbolicEngine):
        self.symbolic = symbolic_engine
        self.equations = {}
        self.parameters = {}
        self.initial_conditions = {}
        self.constraints = []
        self.state_vars = []
        self.coordinates = []
        self.use_hamiltonian = False
        self.hamiltonian_equations = None

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
        """Compile symbolic equations to numerical functions"""
        
        state_vars = []
        for q in coordinates:
            state_vars.extend([q, f"{q}_dot"])
            
        param_subs = {self.symbolic.get_symbol(k): v for k, v in self.parameters.items()}
        
        compiled_equations = {}
        
        for q in coordinates:
            accel_key = f"{q}_ddot"
            if accel_key in accelerations:
                eq = accelerations[accel_key].subs(param_subs)
                
                try:
                    eq = sp.simplify(eq)
                except:
                    pass

                eq = self._replace_derivatives(eq, coordinates)
                
                free_symbols = eq.free_symbols
                ordered_symbols = []
                symbol_indices = []
                
                for i, var_name in enumerate(state_vars):
                    sym = self.symbolic.get_symbol(var_name)
                    if sym in free_symbols:
                        ordered_symbols.append(sym)
                        symbol_indices.append(i)
                
                if ordered_symbols:
                    try:
                        func = sp.lambdify(ordered_symbols, eq, modules=['numpy', 'math'])
                        
                        def make_wrapper(func, indices):
                            def wrapper(*state_vector):
                                try:
                                    args = [state_vector[i] for i in indices if i < len(state_vector)]
                                    if len(args) == len(indices):
                                        result = func(*args)
                                        if isinstance(result, np.ndarray):
                                            result = float(result.item()) if result.size == 1 else float(result[0])
                                        result = float(result)
                                        return result if np.isfinite(result) else 0.0
                                    return 0.0
                                except Exception as e:
                                    return 0.0
                            return wrapper
                        
                        compiled_equations[accel_key] = make_wrapper(func, symbol_indices)
                        
                    except Exception as e:
                        warnings.warn(f"Compilation failed for {accel_key}: {e}")
                        compiled_equations[accel_key] = lambda *args: 0.0
                else:
                    try:
                        const_value = float(sp.N(eq))
                        compiled_equations[accel_key] = lambda *args: const_value
                    except:
                        compiled_equations[accel_key] = lambda *args: 0.0

        self.equations = compiled_equations
        self.state_vars = state_vars
        self.coordinates = coordinates

    def compile_hamiltonian_equations(self, q_dots: List[sp.Expr], p_dots: List[sp.Expr], 
                                     coordinates: List[str]):
        """Compile Hamiltonian equations"""
        self.use_hamiltonian = True
        
        state_vars = []
        for q in coordinates:
            state_vars.extend([q, f"p_{q}"])
        
        param_subs = {self.symbolic.get_symbol(k): v for k, v in self.parameters.items()}
        
        self.hamiltonian_equations = {
            'q_dots': [],
            'p_dots': []
        }
        
        for i, q in enumerate(coordinates):
            q_dot_eq = q_dots[i].subs(param_subs)
            p_dot_eq = p_dots[i].subs(param_subs)
            
            # Compile q_dot
            free_syms = q_dot_eq.free_symbols
            ordered_syms = []
            indices = []
            for j, var_name in enumerate(state_vars):
                sym = self.symbolic.get_symbol(var_name)
                if sym in free_syms:
                    ordered_syms.append(sym)
                    indices.append(j)
            
            if ordered_syms:
                func = sp.lambdify(ordered_syms, q_dot_eq, modules=['numpy', 'math'])
                self.hamiltonian_equations['q_dots'].append((func, indices))
            else:
                const_val = float(sp.N(q_dot_eq))
                self.hamiltonian_equations['q_dots'].append((lambda *args, v=const_val: v, []))
            
            # Compile p_dot
            free_syms = p_dot_eq.free_symbols
            ordered_syms = []
            indices = []
            for j, var_name in enumerate(state_vars):
                sym = self.symbolic.get_symbol(var_name)
                if sym in free_syms:
                    ordered_syms.append(sym)
                    indices.append(j)
            
            if ordered_syms:
                func = sp.lambdify(ordered_syms, p_dot_eq, modules=['numpy', 'math'])
                self.hamiltonian_equations['p_dots'].append((func, indices))
            else:
                const_val = float(sp.N(p_dot_eq))
                self.hamiltonian_equations['p_dots'].append((lambda *args, v=const_val: v, []))
        
        self.state_vars = state_vars
        self.coordinates = coordinates

    def _replace_derivatives(self, expr: sp.Expr, coordinates: List[str]) -> sp.Expr:
        """Replace Derivative objects with corresponding symbols"""
        derivs = list(expr.atoms(sp.Derivative))
        for d in derivs:
            try:
                base = d.args[0]
                order = 1
                if len(d.args) >= 2:
                    arg2 = d.args[1]
                    if isinstance(arg2, tuple) and len(arg2) >= 2:
                        order = int(arg2[1])
                
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
        """ODE system for numerical integration"""
        
        if self.use_hamiltonian:
            return self._hamiltonian_ode(t, y)
        
        dydt = np.zeros_like(y)
        
        for i in range(len(self.coordinates)):
            if 2*i + 1 < len(y):
                dydt[2*i] = y[2*i + 1]
        
        for i, q in enumerate(self.coordinates):
            accel_key = f"{q}_ddot"
            if accel_key in self.equations and 2*i + 1 < len(dydt):
                try:
                    accel_value = self.equations[accel_key](*y)
                    if np.isfinite(accel_value):
                        dydt[2*i + 1] = accel_value
                    else:
                        dydt[2*i + 1] = 0.0
                except Exception:
                    dydt[2*i + 1] = 0.0
                    
        return dydt

    def _hamiltonian_ode(self, t: float, y: np.ndarray) -> np.ndarray:
        """ODE system for Hamiltonian formulation"""
        dydt = np.zeros_like(y)
        
        for i, q in enumerate(self.coordinates):
            # dq/dt
            func, indices = self.hamiltonian_equations['q_dots'][i]
            try:
                args = [y[j] for j in indices if j < len(y)]
                dydt[2*i] = float(func(*args))
            except:
                dydt[2*i] = 0.0
            
            # dp/dt
            func, indices = self.hamiltonian_equations['p_dots'][i]
            try:
                args = [y[j] for j in indices if j < len(y)]
                dydt[2*i + 1] = float(func(*args))
            except:
                dydt[2*i + 1] = 0.0
        
        return dydt

    def simulate(self, t_span: Tuple[float, float], num_points: int = 1000,
                 method: str = 'RK45', rtol: float = 1e-6, atol: float = 1e-8,
                 detect_stiff: bool = True) -> dict:
        """
        Run numerical simulation with adaptive integration and diagnostics
        
        Args:
            t_span: Time span (t_start, t_end)
            num_points: Number of output points
            method: Integration method
            rtol: Relative tolerance
            atol: Absolute tolerance
            detect_stiff: Whether to detect stiff systems
            
        Returns:
            Dictionary with solution data and metadata
        """
        
        y0 = []
        for q in self.coordinates:
            if self.use_hamiltonian:
                pos_val = self.initial_conditions.get(q, 0.0)
                y0.append(pos_val)
                mom_key = f"p_{q}"
                mom_val = self.initial_conditions.get(mom_key, 0.0)
                y0.append(mom_val)
            else:
                pos_val = self.initial_conditions.get(q, 0.0)
                y0.append(pos_val)
                vel_key = f"{q}_dot"
                vel_val = self.initial_conditions.get(vel_key, 0.0)
                y0.append(vel_val)

        y0 = np.array(y0, dtype=float)
        t_eval = np.linspace(t_span[0], t_span[1], num_points)

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
        
        # Detect if system might be stiff
        is_stiff = False
        if detect_stiff and method == 'RK45':
            try:
                # Try a few steps and check Jacobian eigenvalues
                test_sol = solve_ivp(
                    self.equations_of_motion,
                    (t_span[0], t_span[0] + 0.01),
                    y0,
                    method='RK45',
                    max_step=0.001
                )
                if not test_sol.success:
                    is_stiff = True
                    warnings.warn("System may be stiff. Consider using 'LSODA' or 'Radau' method.")
            except:
                pass
        
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
                'is_stiff': is_stiff,
                'use_hamiltonian': self.use_hamiltonian,
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
    """Enhanced visualization with configurable options"""
    
    def __init__(self, trail_length: int = 150, fps: int = 30):
        self.fig = None
        self.ax = None
        self.animation = None
        self.trail_length = trail_length
        self.fps = fps

    def has_ffmpeg(self) -> bool:
        """Check if ffmpeg is available"""
        import shutil
        return shutil.which('ffmpeg') is not None

    def save_animation_to_file(self, anim: animation.FuncAnimation, 
                               filename: str, fps: int = None, dpi: int = 100) -> bool:
        """Save animation to file"""
        if anim is None:
            return False

        fps = fps or self.fps

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
        """Create animated pendulum visualization"""
        
        if not solution['success']:
            warnings.warn("Cannot animate failed simulation")
            return None
            
        self.setup_3d_plot(f"{system_name.title()} Animation")
        
        t = solution['t']
        y = solution['y']
        coordinates = solution['coordinates']
        
        name = (system_name or '').lower()
        
        if len(coordinates) >= 2 or 'double' in name:
            return self._animate_double_pendulum(t, y, parameters)
        else:
            return self._animate_single_pendulum(t, y, parameters)
    
    def _animate_single_pendulum(self, t: np.ndarray, y: np.ndarray, parameters: dict):
        """Animate single pendulum"""
        theta = y[0]
        l = parameters.get('l', 1.0)
        
        x = l * np.sin(theta)
        y_pos = -l * np.cos(theta)
        z = np.zeros_like(x)
        
        self.ax.set_xlim(-l*1.2, l*1.2)
        self.ax.set_ylim(-l*1.2, l*0.2)
        self.ax.set_zlim(-0.1, 0.1)
        
        line, = self.ax.plot([], [], [], 'o-', linewidth=3, markersize=10, 
                            color='#E63946', label='Pendulum')
        trail, = self.ax.plot([], [], [], '-', alpha=0.4, linewidth=1.5, 
                             color='#457B9D', label='Trail')
        time_text = self.ax.text2D(0.05, 0.95, '', transform=self.ax.transAxes, fontsize=12)
        
        self.ax.legend(loc='upper right')
        
        def animate_frame(frame):
            if frame < len(t):
                line.set_data([0, x[frame]], [0, y_pos[frame]])
                line.set_3d_properties([0, z[frame]])
                
                trail_length = min(frame, self.trail_length)
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
        
        x1 = l1 * np.sin(theta1)
        y1 = -l1 * np.cos(theta1)
        x2 = x1 + l2 * np.sin(theta2)
        y2 = y1 - l2 * np.cos(theta2)
        
        max_reach = l1 + l2
        self.ax.set_xlim(-max_reach*1.1, max_reach*1.1)
        self.ax.set_ylim(-max_reach*1.1, max_reach*0.2)
        self.ax.set_zlim(-0.1, 0.1)
        
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
                line1.set_data([0, x1[frame]], [0, y1[frame]])
                line1.set_3d_properties([0, 0])
                
                line2.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
                line2.set_3d_properties([0, 0])
                
                trail_length = min(frame, self.trail_length)
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
        """Animate harmonic oscillator"""
        
        if not solution['success']:
            warnings.warn("Cannot animate failed simulation")
            return None
        
        t = solution['t']
        y = solution['y']
        
        x = y[0]
        v = y[1] if y.shape[0] > 1 else np.zeros_like(x)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.set_xlim(t[0], t[-1])
        ax1.set_ylim(np.min(x)*1.2, np.max(x)*1.2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (m)')
        ax1.set_title(f'{system_name.title()} - Position vs Time')
        ax1.grid(True, alpha=0.3)
        
        line1, = ax1.plot([], [], 'b-', linewidth=2, label='Position')
        point1, = ax1.plot([], [], 'ro', markersize=8)
        ax1.legend()
        
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
                line1.set_data(t[:frame], x[:frame])
                point1.set_data([t[frame]], [x[frame]])
                
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
        """Generic animation dispatcher"""
        if not solution or not solution.get('success'):
            return None

        coords = solution.get('coordinates', [])
        name = (system_name or '').lower()

        try:
            if 'pendulum' in name or any('theta' in c for c in coords):
                return self.animate_pendulum(solution, parameters, system_name)
            elif 'oscillator' in name or 'spring' in name or (len(coords) == 1 and 'x' in coords):
                return self.animate_oscillator(solution, parameters, system_name)
            else:
                return self._animate_phase_space(solution, system_name)
                
        except Exception as e:
            warnings.warn(f"Animation failed: {e}")
            return None
    
    def _animate_phase_space(self, solution: dict, system_name: str):
        """Generic phase space animation"""
        t = solution['t']
        y = solution['y']
        coords = solution['coordinates']
        
        if len(coords) == 0:
            return None
        
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
        """Plot energy conservation analysis with offset correction"""
        
        if not solution['success']:
            warnings.warn("Cannot plot energy for failed simulation")
            return
        
        t = solution['t']
        y = solution['y']
        coords = solution['coordinates']
        
        KE = np.zeros_like(t)
        PE = np.zeros_like(t)
        
        if 'theta' in coords[0]:
            if len(coords) == 1:
                theta = y[0]
                theta_dot = y[1]
                m = parameters.get('m', 1.0)
                l = parameters.get('l', 1.0)
                g = parameters.get('g', 9.81)
                
                KE = 0.5 * m * l**2 * theta_dot**2
                PE = m * g * l * (1 - np.cos(theta))
                
            elif len(coords) >= 2:
                theta1, theta1_dot = y[0], y[1]
                theta2, theta2_dot = y[2], y[3]
                m1 = parameters.get('m1', 1.0)
                m2 = parameters.get('m2', 1.0)
                l1 = parameters.get('l1', 1.0)
                l2 = parameters.get('l2', 1.0)
                g = parameters.get('g', 9.81)
                
                # Fixed: proper offset for double pendulum PE
                KE1 = 0.5 * m1 * l1**2 * theta1_dot**2
                KE2 = 0.5 * m2 * (l1**2 * theta1_dot**2 + l2**2 * theta2_dot**2 + 
                                  2 * l1 * l2 * theta1_dot * theta2_dot * np.cos(theta1 - theta2))
                KE = KE1 + KE2
                
                # Proper potential energy (relative to pivot)
                PE1 = -m1 * g * l1 * np.cos(theta1)
                PE2 = -m2 * g * (l1 * np.cos(theta1) + l2 * np.cos(theta2))
                PE = PE1 + PE2
                # Add constant offset to make PE=0 at lowest point
                PE_offset = -m1 * g * l1 - m2 * g * (l1 + l2)
                PE = PE - PE_offset
                
        else:
            x = y[0]
            v = y[1] if y.shape[0] > 1 else np.zeros_like(x)
            m = parameters.get('m', 1.0)
            k = parameters.get('k', 1.0)
            
            KE = 0.5 * m * v**2
            PE = 0.5 * k * x**2
        
        E_total = KE + PE
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Energy Analysis', fontsize=16, fontweight='bold')
        
        axes[0, 0].plot(t, KE, 'r-', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Energy (J)')
        axes[0, 0].set_title('Kinetic Energy')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(t, PE, 'b-', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Energy (J)')
        axes[0, 1].set_title('Potential Energy')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(t, E_total, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Energy (J)')
        axes[1, 0].set_title('Total Energy')
        axes[1, 0].grid(True, alpha=0.3)
        
        E_error = (E_total - E_total[0]) / np.abs(E_total[0]) * 100 if E_total[0] != 0 else (E_total - E_total[0])
        axes[1, 1].plot(t, E_error, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Relative Error (%)')
        axes[1, 1].set_title('Energy Conservation Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n{'='*50}")
        print("Energy Conservation Analysis")
        print(f"{'='*50}")
        print(f"Initial Total Energy: {E_total[0]:.6f} J")
        print(f"Final Total Energy:   {E_total[-1]:.6f} J")
        print(f"Energy Change:        {E_total[-1] - E_total[0]:.6e} J")
        if E_total[0] != 0:
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
# COMPLETE PHYSICS COMPILER
# ============================================================================

class PhysicsCompiler:
    """
    Main compiler class - v0.2.0 with Hamiltonian support
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
        
        self.symbolic = SymbolicEngine()
        self.simulator = NumericalSimulator(self.symbolic)
        self.visualizer = MechanicsVisualizer()
        self.unit_system = UnitSystem()
        
        self.compilation_time = None
        self.equations = None
        self.use_hamiltonian_formulation = False

    def compile_dsl(self, dsl_source: str, use_hamiltonian: bool = False) -> dict:
        """
        Complete compilation pipeline
        
        Args:
            dsl_source: DSL source code
            use_hamiltonian: Force Hamiltonian formulation
            
        Returns:
            Compilation result dictionary
        """
        
        start_time = time.time()
        
        try:
            tokens = tokenize(dsl_source)
            parser = MechanicsParser(tokens)
            self.ast = parser.parse()
            
            if parser.errors:
                warnings.warn(f"Parser found {len(parser.errors)} errors")
            
            self.analyze_semantics()
            
            # Determine which formulation to use
            if self.hamiltonian is not None:
                use_hamiltonian = True
            elif use_hamiltonian and self.lagrangian is not None:
                # Convert Lagrangian to Hamiltonian
                coords = self.get_coordinates()
                L_sympy = self.symbolic.ast_to_sympy(self.lagrangian)
                self.hamiltonian_expr = self.symbolic.lagrangian_to_hamiltonian(L_sympy, coords)
                use_hamiltonian = True
            
            if use_hamiltonian:
                equations = self.derive_hamiltonian_equations()
                self.use_hamiltonian_formulation = True
            else:
                equations = self.derive_equations()
                self.use_hamiltonian_formulation = False
            
            self.equations = equations
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
                'formulation': 'Hamiltonian' if use_hamiltonian else 'Lagrangian',
            }
            
        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
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
        """Extract generalized coordinates"""
        coordinates = []
        
        for var_name, var_info in self.variables.items():
            if (var_info['type'] in ['Angle', 'Position', 'Coordinate', 'Length'] or
                var_name in ['theta', 'theta1', 'theta2', 'x', 'y', 'z', 'r', 'phi', 'psi']):
                coordinates.append(var_name)
                
        return coordinates

    def derive_equations(self) -> Dict[str, sp.Expr]:
        """Derive equations using Lagrangian formulation"""
        
        if self.lagrangian is None:
            raise ValueError("No Lagrangian defined")
        
        L_sympy = self.symbolic.ast_to_sympy(self.lagrangian)
        coordinates = self.get_coordinates()
        
        if not coordinates:
            raise ValueError("No generalized coordinates found")
        
        eq_list = self.symbolic.derive_equations_of_motion(L_sympy, coordinates)
        accelerations = self.symbolic.solve_for_accelerations(eq_list, coordinates)
        
        return accelerations

    def derive_hamiltonian_equations(self) -> Tuple[List[sp.Expr], List[sp.Expr]]:
        """Derive equations using Hamiltonian formulation"""
        
        if self.hamiltonian is not None:
            H_sympy = self.symbolic.ast_to_sympy(self.hamiltonian)
        elif hasattr(self, 'hamiltonian_expr'):
            H_sympy = self.hamiltonian_expr
        else:
            raise ValueError("No Hamiltonian defined or derived")
        
        coordinates = self.get_coordinates()
        
        if not coordinates:
            raise ValueError("No generalized coordinates found")
        
        q_dots, p_dots = self.symbolic.derive_hamiltonian_equations(H_sympy, coordinates)
        
        return (q_dots, p_dots)

    def setup_simulation(self, equations):
        """Configure numerical simulator"""
        
        parameters = {}
        
        for param_name, param_info in self.parameters_def.items():
            parameters[param_name] = param_info['value']
        
        for var_name, var_info in self.variables.items():
            if var_info['type'] in ['Real', 'Mass', 'Length', 'Acceleration', 'Spring Constant']:
                if var_name not in parameters:
                    defaults = {
                        'g': 9.81,
                        'm': 1.0, 'm1': 1.0, 'm2': 1.0,
                        'l': 1.0, 'l1': 1.0, 'l2': 1.0,
                        'k': 1.0,
                    }
                    parameters[var_name] = defaults.get(var_name, 1.0)
        
        self.simulator.set_parameters(parameters)
        self.simulator.set_initial_conditions(self.initial_conditions)
        
        coordinates = self.get_coordinates()
        
        if self.use_hamiltonian_formulation:
            q_dots, p_dots = equations
            self.simulator.compile_hamiltonian_equations(q_dots, p_dots, coordinates)
        else:
            self.simulator.compile_equations(equations, coordinates)

    def simulate(self, t_span: Tuple[float, float] = (0, 10), 
                num_points: int = 1000, **kwargs):
        """Run numerical simulation"""
        return self.simulator.simulate(t_span, num_points, **kwargs)

    def animate(self, solution: dict, show: bool = True):
        """Create animation from solution"""
        parameters = self.simulator.parameters
        anim = self.visualizer.animate(solution, parameters, self.system_name)
        
        if show and anim is not None:
            plt.show()
        
        return anim

    def export_animation(self, solution: dict, filename: str, fps: int = 30, dpi: int = 100):
        """Export animation to file"""
        anim = self.animate(solution, show=False)
        
        if anim is None:
            raise RuntimeError('No animation available')
        
        ok = self.visualizer.save_animation_to_file(anim, filename, fps, dpi)
        
        if not ok:
            raise RuntimeError(f'Failed to save animation to {filename}')
        
        return filename

    def plot_energy(self, solution: dict):
        """Plot energy analysis"""
        self.visualizer.plot_energy(solution, self.simulator.parameters, self.lagrangian)

    def plot_phase_space(self, solution: dict, coordinate_index: int = 0):
        """Plot phase space"""
        self.visualizer.plot_phase_space(solution, coordinate_index)
    
    def print_equations(self):
        """Print derived equations"""
        if self.equations is None:
            print("No equations derived yet.")
            return
        
        print(f"\n{'='*70}")
        print(f"Equations of Motion: {self.system_name}")
        print(f"Formulation: {'Hamiltonian' if self.use_hamiltonian_formulation else 'Lagrangian'}")
        print(f"{'='*70}\n")
        
        if self.use_hamiltonian_formulation:
            q_dots, p_dots = self.equations
            coords = self.get_coordinates()
            for i, q in enumerate(coords):
                print(f"d{q}/dt = {q_dots[i]}")
                print(f"dp_{q}/dt = {p_dots[i]}\n")
        else:
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
            'formulation': 'Hamiltonian' if self.use_hamiltonian_formulation else 'Lagrangian',
        }


# ============================================================================
# EXAMPLE SYSTEMS - EXPANDED
# ============================================================================

def example_simple_pendulum() -> str:
    """Example: Simple pendulum"""
    return r"""
\system{simple_pendulum}

\defvar{theta}{Angle}{rad}
\defvar{m}{Mass}{kg}
\defvar{l}{Length}{m}
\defvar{g}{Acceleration}{m/s^2}

\parameter{m}{1.0}{kg}
\parameter{l}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

\lagrangian{\frac{1}{2} * m * l^2 * \dot{theta}^2 - m * g * l * (1 - \cos{theta})}

\initial{theta=0.5, theta_dot=0.0}

\solve{RK45}
\animate{pendulum}
"""

def example_double_pendulum() -> str:
    """Example: Double pendulum (chaotic)"""
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
    \frac{1}{2} * (m1 + m2) * l1^2 * \dot{theta1}^2 
    + \frac{1}{2} * m2 * l2^2 * \dot{theta2}^2 
    + m2 * l1 * l2 * \dot{theta1} * \dot{theta2} * \cos{theta1 - theta2}
    + (m1 + m2) * g * l1 * \cos{theta1}
    + m2 * g * l2 * \cos{theta2}
}

\initial{theta1=1.57, theta1_dot=0.0, theta2=1.57, theta2_dot=0.0}

\solve{RK45}
\animate{double_pendulum}
"""

def example_harmonic_oscillator() -> str:
    """Example: Harmonic oscillator"""
    return r"""
\system{harmonic_oscillator}

\defvar{x}{Position}{m}
\defvar{m}{Mass}{kg}
\defvar{k}{Spring Constant}{N/m}

\parameter{m}{1.0}{kg}
\parameter{k}{10.0}{N/m}

\lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}

\initial{x=1.0, x_dot=0.0}

\solve{RK45}
\animate{oscillator}
"""

def example_damped_oscillator() -> str:
    """Example: Damped harmonic oscillator (NEW)"""
    return r"""
\system{damped_oscillator}

\defvar{x}{Position}{m}
\defvar{m}{Mass}{kg}
\defvar{k}{Spring Constant}{N/m}
\defvar{c}{Damping}{kg/s}

\parameter{m}{1.0}{kg}
\parameter{k}{10.0}{N/m}
\parameter{c}{0.5}{kg/s}

\lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2 - \frac{1}{2} * c * \dot{x}^2}

\initial{x=1.0, x_dot=0.0}

\solve{RK45}
\animate{oscillator}
"""

def example_coupled_oscillators() -> str:
    """Example: Coupled harmonic oscillators (NEW)"""
    return r"""
\system{coupled_oscillators}

\defvar{x1}{Position}{m}
\defvar{x2}{Position}{m}
\defvar{m}{Mass}{kg}
\defvar{k1}{Spring Constant}{N/m}
\defvar{k2}{Spring Constant}{N/m}
\defvar{k_c}{Coupling Constant}{N/m}

\parameter{m}{1.0}{kg}
\parameter{k1}{10.0}{N/m}
\parameter{k2}{10.0}{N/m}
\parameter{k_c}{2.0}{N/m}

\lagrangian{
    \frac{1}{2} * m * \dot{x1}^2 + \frac{1}{2} * m * \dot{x2}^2
    - \frac{1}{2} * k1 * x1^2 - \frac{1}{2} * k2 * x2^2
    - \frac{1}{2} * k_c * (x2 - x1)^2
}

\initial{x1=1.0, x1_dot=0.0, x2=0.0, x2_dot=0.0}

\solve{RK45}
"""

def example_spherical_pendulum() -> str:
    """Example: Spherical pendulum (NEW)"""
    return r"""
\system{spherical_pendulum}

\defvar{theta}{Angle}{rad}
\defvar{phi}{Angle}{rad}
\defvar{m}{Mass}{kg}
\defvar{l}{Length}{m}
\defvar{g}{Acceleration}{m/s^2}

\parameter{m}{1.0}{kg}
\parameter{l}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

\lagrangian{
    \frac{1}{2} * m * l^2 * (\dot{theta}^2 + \sin{theta}^2 * \dot{phi}^2)
    + m * g * l * \cos{theta}
}

\initial{theta=0.5, theta_dot=0.0, phi=0.0, phi_dot=2.0}

\solve{RK45}
"""

def run_example(example_name: str = "simple_pendulum", 
                t_span: Tuple[float, float] = (0, 10),
                show_animation: bool = True,
                show_energy: bool = True,
                show_phase: bool = True,
                use_hamiltonian: bool = False) -> dict:
    """
    Run a built-in example system
    
    Args:
        example_name: Name of example
        t_span: Time span for simulation
        show_animation: Whether to show animation
        show_energy: Whether to show energy plot
        show_phase: Whether to show phase space plot
        use_hamiltonian: Use Hamiltonian formulation
        
    Returns:
        Dictionary with compiler and solution
    """
    
    examples = {
        'simple_pendulum': example_simple_pendulum(),
        'double_pendulum': example_double_pendulum(),
        'harmonic_oscillator': example_harmonic_oscillator(),
        'damped_oscillator': example_damped_oscillator(),
        'coupled_oscillators': example_coupled_oscillators(),
        'spherical_pendulum': example_spherical_pendulum(),
    }
    
    if example_name not in examples:
        raise ValueError(f"Unknown example: {example_name}. Choose from {list(examples.keys())}")
    
    dsl_code = examples[example_name]
    
    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(dsl_code, use_hamiltonian=use_hamiltonian)
    
    if not result['success']:
        print(f"Compilation failed: {result.get('error', 'Unknown error')}")
        if 'traceback' in result:
            print(result['traceback'])
        return {'compiler': compiler, 'solution': None}
    
    print(f"\n{'='*70}")
    print(f"Successfully compiled: {result['system_name']}")
    print(f"Formulation: {result['formulation']}")
    print(f"Coordinates: {result['coordinates']}")
    print(f"Compilation time: {result['compilation_time']:.4f} seconds")
    print(f"{'='*70}\n")
    
    compiler.print_equations()
    
    print("Running simulation...")
    solution = compiler.simulate(t_span, num_points=1000)
    
    if not solution['success']:
        print(f"Simulation failed: {solution.get('error', 'Unknown error')}")
        return {'compiler': compiler, 'solution': solution}
    
    print(f"Simulation completed: {solution['nfev']} function evaluations")
    if solution.get('is_stiff'):
        print("⚠️  System detected as potentially stiff")
    
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
# VALIDATION AND TESTING - ENHANCED
# ============================================================================

class SystemValidator:
    """Validate DSL systems against known analytical solutions"""
    
    @staticmethod
    def validate_simple_harmonic_oscillator(compiler: PhysicsCompiler, 
                                           solution: dict,
                                           tolerance: float = 0.01) -> bool:
        """Validate harmonic oscillator against analytical solution"""
        if not solution['success']:
            return False
        
        t = solution['t']
        x = solution['y'][0]
        v = solution['y'][1]
        
        m = compiler.simulator.parameters.get('m', 1.0)
        k = compiler.simulator.parameters.get('k', 1.0)
        
        omega = np.sqrt(k / m)
        
        x0 = x[0]
        v0 = v[0]
        A = np.sqrt(x0**2 + (v0/omega)**2)
        phi = np.arctan2(-v0/omega, x0)
        
        x_analytical = A * np.cos(omega * t + phi)
        
        error = np.max(np.abs(x - x_analytical)) / (A if A != 0 else 1.0)
        
        print(f"\n{'='*50}")
        print("Harmonic Oscillator Validation")
        print(f"{'='*50}")
        print(f"  Natural frequency: {omega:.4f} rad/s")
        print(f"  Amplitude: {A:.4f} m")
        print(f"  Max relative error: {error:.6f}")
        print(f"  Tolerance: {tolerance}")
        print(f"  Status: {'✓ PASSED' if error < tolerance else '✗ FAILED'}")
        print(f"{'='*50}\n")
        
        return error < tolerance
    
    @staticmethod
    def validate_energy_conservation(compiler: PhysicsCompiler,
                                    solution: dict,
                                    tolerance: float = 0.01) -> bool:
        """Validate energy conservation"""
        if not solution['success']:
            return False
        
        t = solution['t']
        y = solution['y']
        coords = solution['coordinates']
        params = compiler.simulator.parameters
        
        KE = np.zeros_like(t)
        PE = np.zeros_like(t)
        
        if 'theta' in coords[0]:
            if len(coords) == 1:
                theta = y[0]
                theta_dot = y[1]
                m = params.get('m', 1.0)
                l = params.get('l', 1.0)
                g = params.get('g', 9.81)
                
                KE = 0.5 * m * l**2 * theta_dot**2
                PE = m * g * l * (1 - np.cos(theta))
            elif len(coords) >= 2:
                theta1, theta1_dot = y[0], y[1]
                theta2, theta2_dot = y[2], y[3]
                m1 = params.get('m1', 1.0)
                m2 = params.get('m2', 1.0)
                l1 = params.get('l1', 1.0)
                l2 = params.get('l2', 1.0)
                g = params.get('g', 9.81)
                
                KE1 = 0.5 * m1 * l1**2 * theta1_dot**2
                KE2 = 0.5 * m2 * (l1**2 * theta1_dot**2 + l2**2 * theta2_dot**2 + 
                                  2 * l1 * l2 * theta1_dot * theta2_dot * np.cos(theta1 - theta2))
                KE = KE1 + KE2
                
                PE1 = -m1 * g * l1 * np.cos(theta1)
                PE2 = -m2 * g * (l1 * np.cos(theta1) + l2 * np.cos(theta2))
                PE = PE1 + PE2
                PE_offset = -m1 * g * l1 - m2 * g * (l1 + l2)
                PE = PE - PE_offset
        else:
            x = y[0]
            v = y[1]
            m = params.get('m', 1.0)
            k = params.get('k', 1.0)
            
            KE = 0.5 * m * v**2
            PE = 0.5 * k * x**2
        
        E_total = KE + PE
        E_error = np.abs((E_total - E_total[0]) / (E_total[0] if E_total[0] != 0 else 1.0))
        max_error = np.max(E_error)
        
        print(f"\n{'='*50}")
        print("Energy Conservation Validation")
        print(f"{'='*50}")
        print(f"  Initial energy: {E_total[0]:.6f} J")
        print(f"  Final energy: {E_total[-1]:.6f} J")
        print(f"  Max relative error: {max_error:.6f}")
        print(f"  Tolerance: {tolerance}")
        print(f"  Status: {'✓ PASSED' if max_error < tolerance else '✗ FAILED'}")
        print(f"{'='*50}\n")
        
        return max_error < tolerance

    @staticmethod
    def run_all_tests() -> dict:
        """Run comprehensive test suite"""
        print("\n" + "="*70)
        print("MechanicsDSL v0.2.0 - Comprehensive Test Suite")
        print("="*70 + "\n")
        
        results = {}
        
        # Test 1: Simple pendulum
        print("Test 1: Simple Pendulum")
        print("-" * 50)
        try:
            output = run_example('simple_pendulum', t_span=(0, 5), 
                               show_animation=False, show_energy=False, show_phase=False)
            compiler = output['compiler']
            solution = output['solution']
            results['simple_pendulum'] = {
                'compiled': output['result']['success'],
                'simulated': solution['success']
            }
        except Exception as e:
            print(f"✗ Failed: {e}")
            results['simple_pendulum'] = {'compiled': False, 'simulated': False}
        
        # Test 2: Harmonic oscillator with validation
        print("\nTest 2: Harmonic Oscillator (with validation)")
        print("-" * 50)
        try:
            output = run_example('harmonic_oscillator', t_span=(0, 10),
                               show_animation=False, show_energy=False, show_phase=False)
            compiler = output['compiler']
            solution = output['solution']
            
            validator = SystemValidator()
            passed = validator.validate_simple_harmonic_oscillator(compiler, solution)
            
            results['harmonic_oscillator'] = {
                'compiled': output['result']['success'],
                'simulated': solution['success'],
                'validated': passed
            }
        except Exception as e:
            print(f"✗ Failed: {e}")
            results['harmonic_oscillator'] = {'compiled': False, 'simulated': False, 'validated': False}
        
        # Test 3: Double pendulum
        print("\nTest 3: Double Pendulum (Chaotic System)")
        print("-" * 50)
        try:
            output = run_example('double_pendulum', t_span=(0, 5),
                               show_animation=False, show_energy=False, show_phase=False)
            compiler = output['compiler']
            solution = output['solution']
            
            validator = SystemValidator()
            energy_ok = validator.validate_energy_conservation(compiler, solution, tolerance=0.05)
            
            results['double_pendulum'] = {
                'compiled': output['result']['success'],
                'simulated': solution['success'],
                'energy_conserved': energy_ok
            }
        except Exception as e:
            print(f"✗ Failed: {e}")
            results['double_pendulum'] = {'compiled': False, 'simulated': False}
        
        # Test 4: Coupled oscillators (NEW)
        print("\nTest 4: Coupled Oscillators")
        print("-" * 50)
        try:
            output = run_example('coupled_oscillators', t_span=(0, 10),
                               show_animation=False, show_energy=False, show_phase=False)
            results['coupled_oscillators'] = {
                'compiled': output['result']['success'],
                'simulated': output['solution']['success']
            }
        except Exception as e:
            print(f"✗ Failed: {e}")
            results['coupled_oscillators'] = {'compiled': False, 'simulated': False}
        
        # Summary
        print("\n" + "="*70)
        print("Test Summary")
        print("="*70)
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, test_results in results.items():
            print(f"\n{test_name}:")
            for key, value in test_results.items():
                status = "✓" if value else "✗"
                print(f"  {status} {key}: {value}")
                total_tests += 1
                if value:
                    passed_tests += 1
        
        print(f"\n{'='*70}")
        print(f"Overall: {passed_tests}/{total_tests} tests passed")
        print(f"{'='*70}\n")
        
        return results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command line interface for MechanicsDSL"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='MechanicsDSL v0.2.0 - Domain-Specific Language for Classical Mechanics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run built-in example
  python mechanics_dsl.py --example simple_pendulum
  
  # Use Hamiltonian formulation
  python mechanics_dsl.py --example simple_pendulum --hamiltonian
  
  # Run comprehensive tests
  python mechanics_dsl.py --test
  
  # Compile and simulate custom DSL file
  python mechanics_dsl.py --file my_system.dsl --time 20 --export animation.mp4
        """
    )
    
    parser.add_argument('--example', type=str, 
                       choices=['simple_pendulum', 'double_pendulum', 'harmonic_oscillator',
                               'damped_oscillator', 'coupled_oscillators', 'spherical_pendulum'],
                       help='Run a built-in example system')
    parser.add_argument('--file', type=str, help='DSL source file to compile')
    parser.add_argument('--time', type=float, default=10.0, help='Simulation time (default: 10.0)')
    parser.add_argument('--points', type=int, default=1000, help='Number of time points (default: 1000)')
    parser.add_argument('--export', type=str, help='Export animation to file (.mp4 or .gif)')
    parser.add_argument('--energy', action='store_true', help='Show energy analysis')
    parser.add_argument('--phase', action='store_true', help='Show phase space plot')
    parser.add_argument('--validate', action='store_true', help='Run validation tests')
    parser.add_argument('--no-animation', action='store_true', help='Skip animation display')
    parser.add_argument('--hamiltonian', action='store_true', help='Use Hamiltonian formulation')
    parser.add_argument('--test', action='store_true', help='Run comprehensive test suite')
    
    args = parser.parse_args()
    
    if args.test:
        SystemValidator.run_all_tests()
        return 0
    
    if args.example:
        results = run_example(
            args.example,
            t_span=(0, args.time),
            show_animation=not args.no_animation,
            show_energy=args.energy,
            show_phase=args.phase,
            use_hamiltonian=args.hamiltonian
        )
        
        compiler = results['compiler']
        solution = results['solution']
        
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                dsl_code = f.read()
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found")
            return 1
        
        compiler = PhysicsCompiler()
        result = compiler.compile_dsl(dsl_code, use_hamiltonian=args.hamiltonian)
        
        if not result['success']:
            print(f"Compilation failed: {result.get('error', 'Unknown error')}")
            if 'traceback' in result:
                print(result['traceback'])
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
    
    if args.export and solution and solution['success']:
        print(f"\nExporting animation to {args.export}...")
        try:
            compiler.export_animation(solution, args.export)
            print(f"Animation saved successfully!")
        except Exception as e:
            print(f"Export failed: {e}")
    
    if args.validate and solution and solution['success']:
        print("\nRunning validation...")
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
    'example_damped_oscillator',
    'example_coupled_oscillators',
    'example_spherical_pendulum',
    'run_example',
]


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) == 1:
        print("""
╔═══════════════════════════════════════════════════════════════════╗
║                      MechanicsDSL v0.2.0                          ║
║       A Domain-Specific Language for Classical Mechanics          ║
║                                                                   ║
║  NEW in v0.2.0:                                                   ║
║  • Hamiltonian formulation support                                ║
║  • Improved parser with better error handling                     ║
║  • Enhanced energy conservation analysis                          ║
║  • New example systems (coupled oscillators, etc.)                ║
║  • Comprehensive test suite                                       ║
║  • Better numerical stability detection                           ║
╚═══════════════════════════════════════════════════════════════════╝

Running interactive demo with simple pendulum...
        """)
        
        results = run_example('simple_pendulum', t_span=(0, 10))
        
        print("\n" + "="*70)
        print("Demo completed! Try these options:")
        print("  --example double_pendulum    # See chaotic behavior")
        print("  --hamiltonian               # Use Hamiltonian formulation")
        print("  --test                      # Run full test suite")
        print("  --help                      # See all options")
        print("="*70)
    else:
        sys.exit(main())
