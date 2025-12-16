"""
A pytest module to test the accuracy of FieldArray bases and basis conversion.
"""

# import numpy as np
# import pytest

# import galois


# @pytest.mark.parametrize("p, m", [(2, 3), (3, 2), (5, 2)])
# def test_polynomial_basis_power_structure(p, m):
#     """
#     GF(p^m): polynomial_basis(order="asc") should be (1, α, α^2, ..., α^{m-1})
#     for some α = image of x.
#     """
#     GF = galois.GF(p**m)
#     basis_asc = GF.polynomial_basis(order="asc")

#     # Length m
#     assert basis_asc.size == m

#     # First element is 1
#     assert np.all(basis_asc[0] == GF(1))

#     # Basis is successive powers of basis[1]
#     alpha = basis_asc[1]
#     for i in range(m):
#         assert np.all(basis_asc[i] == alpha**i)


# @pytest.mark.parametrize("p, m", [(2, 3), (3, 2), (5, 2)])
# def test_polynomial_basis_ordering(p, m):
#     """
#     order="desc" should just reverse order="asc".
#     """
#     GF = galois.GF(p**m)
#     basis_asc = GF.polynomial_basis(order="asc")
#     basis_desc = GF.polynomial_basis(order="desc")

#     assert np.array_equal(basis_desc, basis_asc[::-1])


# @pytest.mark.parametrize("p, m", [(2, 3), (3, 2), (5, 2)])
# def test_normal_element_and_basis_are_consistent(p, m):
#     """
#     For non-prime fields GF(p^m), normal_element should be scalar and
#     normal_basis(None) should generate its Frobenius conjugates.
#     """
#     GF = galois.GF(p**m)
#     # Only test proper extensions
#     if GF.is_prime_field:
#         pytest.skip("Normal basis is not defined for prime fields.")

#     beta = GF.normal_element
#     assert beta.ndim == 0

#     basis_asc = GF.normal_basis(order="asc")
#     assert basis_asc.size == m

#     # First element is β
#     assert np.all(basis_asc[0] == beta)

#     # Successive elements are Frobenius powers β^(p^i)
#     q = GF.characteristic  # for your current implementation this is p
#     for i in range(1, m):
#         assert np.all(basis_asc[i] == basis_asc[i - 1] ** q)

#     # Basis elements should be linearly independent over GF(p):
#     # use .vector() to get coordinates and rank check as proxy
#     M = basis_asc.vector()  # shape (m, m)
#     rank = np.linalg.matrix_rank(M)
#     assert rank == m


# @pytest.mark.parametrize("p", [2, 3, 5, 7])
# def test_normal_element_prime_field_behavior(p):
#     """
#     In a prime field GF(p), normal_element should be None
#     and normal_elements should be empty.
#     """
#     GF = galois.GF(p)
#     assert GF.is_prime_field

#     assert GF.normal_element is None
#     assert GF.normal_elements.size == 0


# @pytest.mark.parametrize("p, m", [(2, 3), (3, 2), (5, 2)])
# def test_change_of_basis_matrix_invertibility(p, m):
#     """
#     Polynomial basis <-> normal basis change of basis matrices
#     should be inverses of each other (up to numerical tolerance).
#     """
#     GF = galois.GF(p**m)
#     if GF.is_prime_field:
#         pytest.skip("Normal basis not defined for prime fields.")

#     poly_basis = GF.polynomial_basis(order="asc")
#     normal_basis = GF.normal_basis(order="asc")

#     T_poly_to_normal = GF.change_of_basis_matrix(poly_basis, normal_basis)
#     T_normal_to_poly = GF.change_of_basis_matrix(normal_basis, poly_basis)

#     # T_normal_to_poly * T_poly_to_normal ≈ I and vice versa
#     I = np.eye(m)
#     assert np.array_equal(T_normal_to_poly @ T_poly_to_normal, I)
#     assert np.array_equal(T_poly_to_normal @ T_normal_to_poly, I)


# @pytest.mark.parametrize("p, m", [(2, 3), (3, 2), (5, 2)])
# def test_change_of_basis_matrix_maps_known_basis(p, m):
#     """
#     Check that the change_of_basis_matrix really maps basis vectors correctly:
#     coordinates of from_basis in to_basis should be the columns of T.
#     """
#     GF = galois.GF(p**m)
#     if GF.is_prime_field:
#         pytest.skip("Normal basis not defined for prime fields.")

#     B = GF.polynomial_basis(order="asc")
#     C = GF.normal_basis(order="asc")

#     T_B_to_C = GF.change_of_basis_matrix(B, C)

#     # Each basis vector b_i expressed in basis C should give unit vectors e_i
#     # when we apply the inverse transform.
#     T_C_to_B = GF.change_of_basis_matrix(C, B)

#     # Coordinates of C in C are identity; coordinates of C in B are T_C_to_B.
#     I = np.eye(B.size)
#     assert np.array_equal(T_C_to_B @ T_B_to_C, I)


# @pytest.mark.parametrize("p, m", [(2, 3), (3, 2), (5, 2)])
# def test_change_bases_round_trip(p, m):
#     """
#     Changing bases polynomial -> normal -> polynomial should be identity.
#     """
#     GF = galois.GF(p**m)
#     if GF.is_prime_field:
#         pytest.skip("Normal basis not defined for prime fields.")

#     poly_basis = GF.polynomial_basis(order="asc")
#     normal_basis = GF.normal_basis(order="asc")

#     # Use a bunch of elements, not just scalars
#     x = GF.elements  # shape (q^m,)

#     x_in_normal = x.change_bases(poly_basis, normal_basis)
#     x_back = x_in_normal.change_bases(normal_basis, poly_basis)

#     assert np.array_equal(x, x_back)


# @pytest.mark.parametrize("p, m", [(2, 3), (3, 2)])
# def test_change_bases_preserves_shape(p, m):
#     """
#     change_bases should act elementwise and preserve the shape of the array.
#     """
#     GF = galois.GF(p**m)
#     if GF.is_prime_field:
#         pytest.skip("Normal basis not defined for prime fields.")

#     poly_basis = GF.polynomial_basis(order="asc")
#     normal_basis = GF.normal_basis(order="asc")

#     x = GF.Random((4, 5))  # 2D array of elements

#     y = x.change_bases(poly_basis, normal_basis)
#     z = y.change_bases(normal_basis, poly_basis)

#     assert y.shape == x.shape
#     assert z.shape == x.shape
#     assert np.array_equal(x, z)
