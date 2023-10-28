module TestMOI

using Test
import MathOptInterface as MOI
import NLPModelsJuMP
import Percival

function test_runtests()
  model = MOI.instantiate(NLPModelsJuMP.Optimizer, with_bridge_type = Float64)
  MOI.set(model, MOI.RawOptimizerAttribute("solver"), Percival.PercivalSolver)
  MOI.set(model, MOI.Silent(), true) # comment this to enable output
  config = MOI.Test.Config(
    atol = 1e-2,
    optimal_status = MOI.LOCALLY_SOLVED,
    exclude = Any[
      MOI.ConstraintBasisStatus,
      MOI.VariableBasisStatus,
      MOI.ConstraintName,
      MOI.VariableName,
      MOI.ObjectiveBound,
      MOI.SolverVersion,
      # TODO dual not done yet
      MOI.DualObjectiveValue,
      MOI.ConstraintDual,
    ],
  )
  MOI.Test.runtests(
    model,
    config,
    exclude = [
      # test_solve_TerminationStatus_DUAL_INFEASIBLE:
      #  Expression: status in (MOI.DUAL_INFEASIBLE, MOI.NORM_LIMIT, MOI.INFEASIBLE_OR_UNBOUNDED)
      #   Evaluated: MathOptInterface.ITERATION_LIMIT in (MathOptInterface.DUAL_INFEASIBLE, MathOptInterface.NORM_LIMIT, MathOptInterface.INFEASIBLE_OR_UNBOUNDED)
      r"test_solve_TerminationStatus_DUAL_INFEASIBLE$",
      # AssertionError: feat in MOI.features_available(d)
      r"test_nonlinear_invalid$",
      # Unsupported feature JacVec
      r"test_nonlinear_objective$",
      r"test_nonlinear_objective_and_moi_objective_test$",
      r"test_nonlinear_without_objective$",
      r"test_nonlinear_hs071_NLPBlockDual$",
      r"test_nonlinear_hs071_no_hessian$",
      r"test_nonlinear_hs071_hessian_vector_product$",
      r"test_nonlinear_hs071$",
      # FIXME We should look over all attributes set and error
      #       for unknown ones instead of just getting the ones we know we support
      #       and ignoring the rest
      r"test_model_copy_to_UnsupportedAttribute$",
      # FIXME investigate
      r"test_modification_transform_singlevariable_lessthan$",
      r"test_modification_set_singlevariable_lessthan$",
      r"test_linear_DUAL_INFEASIBLE_2$",
      r"test_quadratic_duplicate_terms$",
      r"test_modification_set_scalaraffine_lessthan$",
      r"test_modification_multirow_vectoraffine_nonpos$",
      r"test_linear_INFEASIBLE$",
      r"test_variable_solve_with_upperbound$",
      r"test_quadratic_nonhomogeneous$",
      r"test_modification_func_scalaraffine_lessthan$",
      r"test_modification_affine_deletion_edge_cases$",
      r"test_modification_coef_scalar_objective$",
      r"test_modification_coef_scalaraffine_lessthan$",
      r"test_modification_const_scalar_objective$",
      r"test_modification_const_vectoraffine_nonpos$",
      r"test_conic_NormInfinityCone_3$",
      r"test_conic_NormInfinityCone_INFEASIBLE$",
      r"test_conic_NormInfinityCone_VectorAffineFunction$",
      r"test_conic_NormInfinityCone_VectorOfVariables$",
      r"test_conic_NormOneCone$",
      r"test_conic_NormOneCone_INFEASIBLE$",
      r"test_conic_NormOneCone_VectorAffineFunction$",
      r"test_conic_NormOneCone_VectorOfVariables$",
      r"test_conic_linear_INFEASIBLE$",
      r"test_conic_linear_INFEASIBLE_2$",
      r"test_conic_linear_VectorAffineFunction$",
      r"test_conic_linear_VectorAffineFunction_2$",
      r"test_conic_linear_VectorOfVariables$",
      r"test_conic_linear_VectorOfVariables_2$",
      r"test_constraint_ScalarAffineFunction_Interval$",
      r"test_constraint_ScalarAffineFunction_LessThan$",
      r"test_constraint_ScalarAffineFunction_duplicate$",
      r"test_constraint_VectorAffineFunction_duplicate$",
      r"test_linear_DUAL_INFEASIBLE$",
      r"test_linear_HyperRectangle_VectorAffineFunction$",
      r"test_linear_HyperRectangle_VectorOfVariables$",
      r"test_linear_INFEASIBLE_2$",
      r"test_linear_Interval_inactive$",
      r"test_linear_LessThan_and_GreaterThan$",
      r"test_linear_VariablePrimalStart_partial$",
      r"test_linear_VectorAffineFunction$",
      r"test_linear_add_constraints$",
      r"test_linear_complex_Zeros$",
      r"test_linear_complex_Zeros_duplicate$",
      r"test_linear_inactive_bounds$",
      r"test_linear_integration$",
      r"test_linear_integration_2$",
      r"test_linear_integration_Interval$",
      r"test_linear_integration_delete_variables$",
      r"test_linear_integration_modification$",
      r"test_linear_modify_GreaterThan_and_LessThan_constraints$",
      r"test_linear_open_intervals$",
      r"test_linear_transform$",
      r"test_linear_variable_open_intervals$",
      r"test_linear_integration_modification$",
      r"test_quadratic_SecondOrderCone_basic$",
      r"test_quadratic_constraint_GreaterThan$",
      r"test_quadratic_constraint_LessThan$",
      r"test_quadratic_constraint_basic$",
      r"test_quadratic_constraint_integration$",
      r"test_quadratic_constraint_minimize$",
      r"test_quadratic_integration$",
      r"test_quadratic_nonconvex_constraint_basic$",
      r"test_quadratic_nonconvex_constraint_integration$",
      r"test_basic_VectorNonlinearFunction_HyperRectangle$",
      r"test_basic_VectorNonlinearFunction_Nonnegatives$",
      r"test_basic_VectorNonlinearFunction_Nonpositives$",
      r"test_basic_VectorNonlinearFunction_Zeros$",
      r"test_constraint_qcp_duplicate_off_diagonal$",
      r"test_nonlinear_expression_hs071$",
      r"test_nonlinear_expression_hs109$",
      r"test_nonlinear_expression_overrides_objective$",
      # ITERATION_LIMIT
      r"test_quadratic_constraint_LessThan$",
      r"test_quadratic_constraint_GreaterThan$",
      r"test_nonlinear_expression_hs071_epigraph$",
      # FIXME Segfault
      r"test_linear_integration_delete_variables$",
      # https://github.com/jump-dev/MathOptInterface.jl/issues/2323
      r"test_basic_VectorNonlinearFunction_NormInfinityCone$",
      r"test_basic_VectorNonlinearFunction_NormOneCone$",
    ],
  )
  return
end

function runtests()
  for name in names(@__MODULE__; all = true)
    if startswith("$(name)", "test_")
      @testset "$(name)" begin
        getfield(@__MODULE__, name)()
      end
    end
  end
  return
end

end  # module

TestMOI.runtests()
