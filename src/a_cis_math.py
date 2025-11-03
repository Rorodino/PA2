"""
CIS PA1 - 3D Cartesian Math Package
Authors: Rohit Satish and Sahana Raja

This module provides 3D mathematical operations for points, rotations, and frame transformations.
Required for electromagnetic tracking system calibration.

Uses NumPy for linear algebra operations.
"""
# Import statements
import numpy as np
from typing import Tuple, List, Optional
import math

# 3D point class - basic operations
class Point3D:

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """Initialize a 3D point with x, y, z coordinates"""
        if not all(isinstance(v, (int, float, np.floating)) for v in (x, y, z)):
            raise TypeError("Point3D coordinates must be numeric (int or float).")
        self.x, self.y, self.z = float(x), float(y), float(z)
    
    def __add__(self, other: 'Point3D') -> 'Point3D':
        """Add two points together"""
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Point3D') -> 'Point3D':
        """Subtract two points"""
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Point3D':
        """Multiply point by scalar"""
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)
      
    def __rmul__(self, scalar: float) -> 'Point3D':
        """Handles the case of the scalar being on the left side of the vector

        Args:
            scalar (float): Scalar value used to scale each component.

        Returns:
            Point3D: Scaled point.
        """
        return self.__mul__(scalar)

    def __eq__(self, other: object) -> bool:
        """Determine whether two points represent the same location in space.

        Args:
            other (object): Object to compare against this point.

        Returns:
            bool: ``True`` when ``other`` is a ``Point3D`` whose ``x``, ``y``,
                and ``z`` coordinates match this point within a tolerance of
                1e-9; ``NotImplemented`` for unsupported types.
        """
        if not isinstance(other, Point3D):
            return NotImplemented
        return (
            math.isclose(self.x, other.x, rel_tol=1e-9, abs_tol=1e-9)
            and math.isclose(self.y, other.y, rel_tol=1e-9, abs_tol=1e-9)
            and math.isclose(self.z, other.z, rel_tol=1e-9, abs_tol=1e-9)
        )

    def dot(self, other: 'Point3D') -> float:
        """Dot Product of two Point3D's

        Args:
            other (Point3D): Point used for the dot product.

        Returns:
            float: Scalar dot product of the two points.
        """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Point3D') -> 'Point3D':
        """Cross Product of two Point3D's

        Args:
            other (Point3D): Point used for the cross product.

        Returns:
            Point3D: Cross product vector of the two points.
        """
        return Point3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def norm(self) -> float:
        """Calculate the Euclidean norm of the point"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> 'Point3D':
        """Returns normalized version of the vector"""
        n = self.norm()
        if n == 0:
            return Point3D(0, 0, 0)
        return Point3D(self.x/n, self.y/n, self.z/n)
    
    def to_array(self) -> np.ndarray:
        """Converts a Point3D to numpy array"""
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Point3D':
        """Create Point3D from numpy array"""
        if len(arr) != 3:
            raise ValueError("Array must have exactly 3 elements")
        return cls(arr[0], arr[1], arr[2])
    
    def __repr__(self) -> str:
        """toString method"""
        return f"Point3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

# 3D Rotation class - handles rotation matricies
class Rotation3D:
    def __init__(self, matrix: Optional[np.ndarray] = None):
        """Initialize the rotation matrix"""
        if matrix is None:
            self.matrix = np.eye(3)  # identity matrix (used this before)
        else:
            candidate = np.asarray(matrix, dtype=float)
            if candidate.shape != (3, 3):
                raise ValueError("Rotation matrix must be 3x3.")
            if not np.allclose(candidate.T @ candidate, np.eye(3), atol=1e-6):
                raise ValueError("Rotation matrix must be orthonormal.")
            if not math.isclose(np.linalg.det(candidate), 1.0, abs_tol=1e-6):
                raise ValueError("Rotation matrix must have determinant +1.")
            self.matrix = candidate.copy()
    
    @classmethod
    def from_axis_angle(cls, axis: Point3D, angle: float) -> 'Rotation3D':
        """Create rotation from axis-angle using Rodrigues formula
        """
        if axis.norm() < 1e-12:
            raise ValueError("Invalid axis: zero-length vector for axis-angle rotation.")
        axis = axis.normalize()
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        one_minus_cos = 1 - cos_angle
        
        # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
        K = np.array([
            [0, -axis.z, axis.y],
            [axis.z, 0, -axis.x],
            [-axis.y, axis.x, 0]
        ])
        
        R = np.eye(3) + sin_angle * K + one_minus_cos * np.dot(K, K)
        return cls(R)
    
    @classmethod
    def from_euler_angles(cls, alpha: float, beta: float, gamma: float, 
                         order: str = 'xyz') -> 'Rotation3D':
        """Create rotation from Euler angles

        Args:
            alpha (float): Rotation angle about the x-axis.
            beta (float): Rotation angle about the y-axis.
            gamma (float): Rotation angle about the z-axis.
            order (str): Order in which rotations are applied.

        Returns:
            Rotation3D: Rotation constructed from Euler angles.
        """
        Rx = cls._rotation_x(alpha)
        Ry = cls._rotation_y(beta)
        Rz = cls._rotation_z(gamma)
        
        if order == 'xyz':
            R = np.dot(Rz, np.dot(Ry, Rx))
        elif order == 'zyz':
            Rz1 = cls._rotation_z(alpha)
            Ry = cls._rotation_y(beta)
            Rz2 = cls._rotation_z(gamma)
            R = Rz2 @ (Ry @ Rz1)
        else:
            raise ValueError(f"Unsupported Euler angle order: {order}")
        
        return cls(R)
    
    @classmethod
    def from_quaternion(cls, q0: float, q1: float, q2: float, q3: float) -> 'Rotation3D':
        """Create rotation from quaternion (w, x, y, z)

        Args:
            q0 (float): Scalar component of the quaternion.
            q1 (float): X component of the quaternion.
            q2 (float): Y component of the quaternion.
            q3 (float): Z component of the quaternion.

        Returns:
            Rotation3D: Rotation constructed from the normalized quaternion.
        """
        # Normalize quaternion
        norm = math.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
        if norm < 1e-12:
            raise ValueError("Invalid quaternion: zero or near-zero norm.")
        q0, q1, q2, q3 = q0/norm, q1/norm, q2/norm, q3/norm
                
        # Convert to rotation matrix
        R = np.array([
            [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2]
        ])
        
        return cls(R)
    
    @staticmethod
    def _rotation_x(angle: float) -> np.ndarray:
        """Rotation matrix about x-axis

        Args:
            angle (float): Rotation angle in radians.

        Returns:
            np.ndarray: Rotation matrix around the x-axis.
        """
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    
    @staticmethod
    def _rotation_y(angle: float) -> np.ndarray:
        """Rotation matrix about y-axis

        Args:
            angle (float): Rotation angle in radians.

        Returns:
            np.ndarray: Rotation matrix around the y-axis.
        """
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    
     #Rotation matrix about z-axis
    @staticmethod
    def _rotation_z(angle: float) -> np.ndarray:
        """Rotation matrix about z-axis

        Args:
            angle (float): Rotation angle in radians.

        Returns:
            np.ndarray: Rotation matrix around the z-axis.
        """
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    
    def apply(self, point: Point3D) -> Point3D:
        """Applies Rotation to a 3D Point so in effect the Rotation class acts upon the Point3D class

        Args:
            point (Point3D): Point to rotate.

        Returns:
            Point3D: Rotated point.
        """
        p_array = point.to_array()
        rotated = np.dot(self.matrix, p_array)
        return Point3D.from_array(rotated)
    
    def inverse(self) -> 'Rotation3D':
        """Applies Inverse Rotation (transpose of rotation matrix)

        Args:
            None: This method uses the rotation matrix of the object.

        Returns:
            Rotation3D: Inverse rotation.
        """
        return Rotation3D(self.matrix.T)
    
    def compose(self, other: 'Rotation3D') -> 'Rotation3D':
        """Compose this rotation with another rotation. Note that this is particularly useful for kinematic chains

        Args:
            other (Rotation3D): Rotation to compose with this rotation.

        Returns:
            Rotation3D: Composition of the rotations.
        """
        return Rotation3D(np.dot(self.matrix, other.matrix))
    
    def determinant(self) -> float:
        """Get determinant of rotation matrix (should be +1). If negative that indicates a reflection is occurring

        Args:
            None: This method uses the rotation matrix of the object.

        Returns:
            float: Determinant of the rotation matrix.
        """
        return np.linalg.det(self.matrix)
    
    def to_axis_angle(self) -> Tuple[Point3D, float]:
        """Convert rotation matrix to axis-angle representation

        Args:
            None: This method uses the rotation matrix of the object.

        Returns:
            Tuple[Point3D, float]: Axis of rotation and rotation angle in radians.
        """
        # Extract axis and angle from rotation matrix
        trace = np.trace(self.matrix)
        angle = math.acos(max(-1, min(1, (trace - 1) / 2)))
        
        if abs(angle) < 1e-6:
            # Identity rotation
            return Point3D(1, 0, 0), 0.0
        
        # Extract axis using numerically stable approach
        axis_vector = Point3D(
            self.matrix[2, 1] - self.matrix[1, 2],
            self.matrix[0, 2] - self.matrix[2, 0],
            self.matrix[1, 0] - self.matrix[0, 1]
        )
        if axis_vector.norm() < 1e-6:
            # Handle the 180-degree rotation case where off-diagonal differences vanish
            R = self.matrix
            axis_components = [
                math.sqrt(max(0.0, (R[0, 0] + 1) / 2)),
                math.sqrt(max(0.0, (R[1, 1] + 1) / 2)),
                math.sqrt(max(0.0, (R[2, 2] + 1) / 2)),
            ]

            # Determine signs using off-diagonal elements
            if axis_components[0] >= axis_components[1] and axis_components[0] >= axis_components[2] and axis_components[0] > 1e-6:
                # Use the dominant diagonal component to infer the axis direction
                axis_components[1] = math.copysign(axis_components[1], R[0, 1])
                axis_components[2] = math.copysign(axis_components[2], R[0, 2])
            elif axis_components[1] >= axis_components[0] and axis_components[1] >= axis_components[2] and axis_components[1] > 1e-6:
                # Same idea, but seeded from the Y-axis component
                axis_components[0] = math.copysign(axis_components[0], R[1, 0])
                axis_components[2] = math.copysign(axis_components[2], R[1, 2])
            elif axis_components[2] > 1e-6:
                # Otherwise fall back to the Z-axis component
                axis_components[0] = math.copysign(axis_components[0], R[2, 0])
                axis_components[1] = math.copysign(axis_components[1], R[2, 1])
            axis_vector = Point3D(*axis_components)

        axis = axis_vector.normalize()
        return axis, angle
    
    def __repr__(self) -> str:
        """toString

        Args:
            None: This method uses the rotation matrix of the object.

        Returns:
            str: String representation of the rotation.
        """
        return f"Rotation3D(\n{self.matrix}\n)"

#3D Frame transformation class combining rotation and translation
class Frame3D:
    
    def __init__(self, rotation: Rotation3D = None, translation: Point3D = None):
        """Initializes the frame transformation with rotation and translation components.

        Args:
            rotation (Rotation3D): Rotation component of the frame.
            translation (Point3D): Translation component of the frame.

        Returns:
            None: This initializer configures the frame in place.
        """
        self.rotation = rotation if rotation is not None else Rotation3D()
        self.translation = translation if translation is not None else Point3D()
    
    @classmethod
    def identity(cls) -> 'Frame3D':
        """Create identity frame transformation

        Args:
            None: Uses default rotation and translation.

        Returns:
            Frame3D: Identity frame with no rotation and zero translation.
        """
        return cls()
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'Frame3D':
        """Create frame from 4x4 homogeneous transformation matrix

        Args:
            matrix (np.ndarray): 4x4 homogeneous transformation matrix.

        Returns:
            Frame3D: Frame constructed from the provided matrix.
        """
        if matrix.shape != (4, 4):
            raise ValueError("Matrix must be 4x4")
        
        rotation = Rotation3D(matrix[:3, :3])
        translation = Point3D(matrix[0, 3], matrix[1, 3], matrix[2, 3])
        return cls(rotation, translation)
    
    def apply(self, point: Point3D) -> Point3D:
        """Apply frame transformation to a 3D point

        Args:
            point (Point3D): Point to transform.

        Returns:
            Point3D: Transformed point.
        """
        rotated = self.rotation.apply(point)
        return rotated + self.translation
    
    def inverse(self) -> 'Frame3D':
        """Get inverse frame transformation

        Args:
            None: This method uses the frame's rotation and translation.

        Returns:
            Frame3D: Inverse of the frame transformation.
        """
        inv_rotation = self.rotation.inverse()
        inv_translation = inv_rotation.apply(Point3D() - self.translation)
        return Frame3D(inv_rotation, inv_translation)
    
    def compose(self, other: 'Frame3D') -> 'Frame3D':
        """Compose this frame with another frame

        Args:
            other (Frame3D): Frame to compose with this frame.

        Returns:
            Frame3D: Composition of the two frames.
        """
        new_rotation = self.rotation.compose(other.rotation)
        new_translation = self.rotation.apply(other.translation) + self.translation
        return Frame3D(new_rotation, new_translation)
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 homogeneous transformation matrix

        Args:
            None: This method uses the frame's rotation and translation.

        Returns:
            np.ndarray: Homogeneous transformation matrix representation of the frame.
        """
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation.matrix
        matrix[:3, 3] = [self.translation.x, self.translation.y, self.translation.z]
        return matrix
    
    def __repr__(self) -> str:
        """toString

        Args:
            None: This method uses the frame's rotation and translation.

        Returns:
            str: String representation of the frame.
        """
        return f"Frame3D(R={self.rotation}, t={self.translation})"

def compute_centroid(points: List[Point3D]) -> Point3D:
    """Compute centroid of a list of 3D points

    Args:
        points (List[Point3D]): Points used to compute the centroid.

    Returns:
        Point3D: Centroid of the provided points.
    """
    if not points:
        return Point3D()
    
    sum_x = sum(p.x for p in points)
    sum_y = sum(p.y for p in points)
    sum_z = sum(p.z for p in points)
    n = len(points)
    
    return Point3D(sum_x/n, sum_y/n, sum_z/n)

def compute_covariance_matrix(points_a: List[Point3D], points_b: List[Point3D]) -> np.ndarray:
    """Compute covariance matrix for point set registration

    Args:
        points_a (List[Point3D]): First set of points.
        points_b (List[Point3D]): Second set of points.

    Returns:
        np.ndarray: Covariance matrix derived from the two point sets.
    """
    if len(points_a) != len(points_b):
        raise ValueError("Point sets must have same length")
    
    n = len(points_a)
    H = np.zeros((3, 3))
    
    for i in range(n):
        a = points_a[i].to_array()
        b = points_b[i].to_array()
        H += np.outer(a, b)
    
    return H

def skew_symmetric_matrix(point: Point3D) -> np.ndarray:
    """Create skew-symmetric matrix from 3D point

    Args:
        point (Point3D): Point from which to construct the matrix.

    Returns:
        np.ndarray: Skew-symmetric matrix representation of the point.
    """
    return np.array([
        [0, -point.z, point.y],
        [point.z, 0, -point.x],
        [-point.y, point.x, 0]
    ])


# Test the implementation
if __name__ == "__main__":
    # Test basic operations
    p1 = Point3D(1, 2, 3)
    p2 = Point3D(4, 5, 6)
    
    print(f"Point 1: {p1}")
    print(f"Point 2: {p2}")
    print(f"Dot product: {p1.dot(p2)}")
    print(f"Cross product: {p1.cross(p2)}")
    print(f"Norm of p1: {p1.norm()}")
    
    # Test rotation
    axis = Point3D(0, 0, 1)  # z-axis
    angle = math.pi / 2  # 90 degrees
    R = Rotation3D.from_axis_angle(axis, angle)
    
    test_point = Point3D(1, 0, 0)
    rotated = R.apply(test_point)
    print(f"Rotated (1,0,0) by 90° around z-axis: {rotated}")
    
    # Test frame transformation
    frame = Frame3D(R, Point3D(1, 1, 1))
    transformed = frame.apply(Point3D(0, 0, 0))
    print(f"Frame transformation of origin: {transformed}")
