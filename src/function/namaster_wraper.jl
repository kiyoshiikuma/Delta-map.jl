using PyCall
@pyimport healpy as hp
@pyimport pymaster as nmt

# ===============================
# Parameters & plan structs
# ===============================

mutable struct NamasterParams
    mask::Vector{Float64}   # input mask at some nside (e.g. same as data or higher)
    nside::Int              # target nside for NaMaster
    lmin::Int
    lmax::Int
    Δℓ::Int
    apodize::Bool
    deg::Real
    method::String
    purify_e::Bool
    purify_b::Bool
    is_Dell::Bool
    from_edges::Bool
end

# convenience outer constructor: only mask + nside are required
function NamasterParams(mask::AbstractVector{<:Real}, nside::Int;
        lmin::Int      = 2,
        lmax::Int      = 2*nside,
        Δℓ::Int        = 10,
        apodize::Bool  = true,
        deg::Real      = 2.0,
        method::String = "C1",
        purify_e::Bool = false,
        purify_b::Bool = true,
        is_Dell::Bool  = true,
        from_edges::Bool = true)

    return NamasterParams(
        Float64.(mask),
        nside,
        lmin,
        lmax,
        Δℓ,
        apodize,
        deg,
        method,
        purify_e,
        purify_b,
        is_Dell,
        from_edges,
    )
end

mutable struct NamasterPlan
    params::NamasterParams
    mask_nm::PyObject   # downgraded + apodized mask (numpy array)
    bin::PyObject       # NmtBin
    leff::Vector{Float64}
    workspace::PyObject # NmtWorkspace (for spin-2 pol)
end

# ===============================
# Small helpers
# ===============================

"Build (ell_ini, ell_end) edges from params."
function create_ell_bins(p::NamasterParams)
    ell_ini = collect(p.lmin:p.Δℓ:p.lmax)
    ell_end = [min(ℓ + p.Δℓ, p.lmax + 1) for ℓ in ell_ini]
    return ell_ini, ell_end
end

"""
Extract (T, Q, U) from common map layouts.

Supported:
- Vector{Float64}(Npix)          -> (T, nothing, nothing)
- 2×Npix matrix (Q/U)            -> (nothing, Q, U)
- 3×Npix matrix (T/Q/U)          -> (T, Q, U)
- Vector{Vector}(2) [Q, U]       -> (nothing, Q, U)
- Vector{Vector}(3) [T, Q, U]    -> (T, Q, U)
"""
function _extract_TQU(map)
    # Matrix case first (2×Npix, 3×Npix, Npix×3)
    if map isa AbstractMatrix
        if size(map, 1) == 3
            return view(map, 1, :), view(map, 2, :), view(map, 3, :)
        elseif size(map, 1) == 2
            return nothing, view(map, 1, :), view(map, 2, :)
        elseif size(map, 2) == 3
            return view(map, :, 1), view(map, :, 2), view(map, :, 3)
        else
            error("Map shape not understood. Use 2×Npix(Q/U) or 3×Npix(T/Q/U).")
        end
    end

    # Vector case
    if map isa AbstractVector
        # Vector of vectors: [Q, U] or [T, Q, U]
        if eltype(map) <: AbstractVector
            if length(map) == 3
                return map[1], map[2], map[3]
            elseif length(map) == 2
                return nothing, map[1], map[2]
            else
                error("Vector-of-vectors map must have length 2 or 3.")
            end
        end
        # Plain vector: treat as T-only (scalar)
        return map, nothing, nothing
    end

    error("map must be a Vector or Matrix.")
end

# ===============================
# Plan construction
# ===============================

"""
Prepare NaMaster plan (spin-2 polarization only).

- Uses params.mask / params.nside / params.lmax / params.Δℓ / etc.
- Pre-computes the coupling matrix with a dummy Q/U map.
"""
function prepare_namaster_plan(params::NamasterParams)
    # downgrade mask to target nside (returns numpy array)
    mask_nm = hp.ud_grade(params.mask, params.nside)

    # optional apodization
    if params.apodize
        mask_nm = nmt.mask_apodization(mask_nm, params.deg; apotype=params.method)
    end

    # define bandpower binning
    if params.from_edges
        ell_ini, ell_end = create_ell_bins(params)
        bin = nmt.NmtBin.from_edges(ell_ini, ell_end; is_Dell=params.is_Dell)
    else
        bin = nmt.NmtBin.from_lmax_linear(params.lmax, params.Δℓ; is_Dell=params.is_Dell)
    end

    leff = Vector{Float64}(bin.get_effective_ells())

    # workspace and coupling matrix (spin-2 pol, dummy map)
    npix = 12 * params.nside^2
    q0 = zeros(npix)
    u0 = zeros(npix)

    field0 = nmt.NmtField(
        mask_nm,
        [q0, u0],
        purify_e = params.purify_e,
        purify_b = params.purify_b,
        lmax     = params.lmax,
        lmax_mask = params.lmax,
    )

    workspace = nmt.NmtWorkspace()
    workspace.compute_coupling_matrix(field0, field0, bin)

    return NamasterPlan(params, mask_nm, bin, leff, workspace)
end

# ===============================
# Map -> binned Cl (EE, BB)
# ===============================

"""
Compute decoupled, binned EE and BB from a Q/U map.

map can be:
- 2×Npix matrix (Q/U)
- 3×Npix matrix (T/Q/U)
- Vector{Vector}(2) [Q, U]
- Vector{Vector}(3) [T, Q, U]

Returns:
- cl_EE::Vector{Float64}
- cl_BB::Vector{Float64}
- leff::Vector{Float64}
"""
function namaster_cl(plan::NamasterPlan, map; map2=nothing)
    nside = plan.params.nside
    npix_expected = 12 * nside^2

    # extract Q/U
    _, q, u = _extract_TQU(map)
    q === nothing && error("This wrapper is spin-2 only: cannot extract Q/U from map.")
    length(q) == npix_expected || error("Q length mismatch (got $(length(q)), expect $(npix_expected))")
    length(u) == npix_expected || error("U length mismatch (got $(length(u)), expect $(npix_expected))")

    if map2 === nothing
        q2, u2 = q, u
    else
        _, q2, u2 = _extract_TQU(map2)
        q2 === nothing && error("Cannot extract Q/U from map2.")
    end

    field1 = nmt.NmtField(
        plan.mask_nm,
        [q, u],
        purify_e = plan.params.purify_e,
        purify_b = plan.params.purify_b,
        lmax     = plan.params.lmax,
        lmax_mask = plan.params.lmax,
    )

    field2 = nmt.NmtField(
        plan.mask_nm,
        [q2, u2],
        purify_e = plan.params.purify_e,
        purify_b = plan.params.purify_b,
        lmax     = plan.params.lmax,
        lmax_mask = plan.params.lmax,
    )

    cl_cpl = nmt.compute_coupled_cell(field1, field2)
    cl_dec = plan.workspace.decouple_cell(cl_cpl)

    # cl_dec is 4×Nb: [EE, EB, BE, BB]
    cl_dec_jl = PyCall.convert(Array{Float64,2}, cl_dec)
    cl_EE = cl_dec_jl[1, :]
    cl_BB = cl_dec_jl[4, :]

    return cl_EE, cl_BB, plan.leff
end

# ===============================
# Theoretical Cl -> binned Cl
# ===============================

"""
Bin a theoretical C_ell (e.g. EE or BB) using the same NaMaster plan.

Input:
- plan :: NamasterPlan
- cl_in :: Vector{<:Real}  (length >= lmax+1, C_ell from ell=0..)

We build a 4×(lmax+1) array:
- EE and BB set to cl_in
- EB=BE=0
and then run couple+decouple to get binned theory.

Returns:
- cl_EE_th_binned::Vector{Float64}
- cl_BB_th_binned::Vector{Float64}
- leff::Vector{Float64}
"""
function calc_cl_th_nm(plan::NamasterPlan, cl_in::AbstractVector{<:Real})
    lmax = plan.params.lmax

    length(cl_in) < lmax + 1 && error("cl_in must have length >= lmax+1 (got $(length(cl_in)), need $(lmax+1))")

    # Julia is 1-based: cl_in[1] corresponds to ell=0
    cl_use = Float64.(cl_in[1:(lmax+1)])

    # Build 4×(lmax+1) theory array in Julia
    cl_th = zeros(Float64, 4, lmax + 1)
    cl_th[1, :] .= cl_use   # EE
    cl_th[4, :] .= cl_use   # BB

    # couple + decouple using the same workspace and binning
    cl_th_cpl = plan.workspace.couple_cell(cl_th)
    cl_th_dec = plan.workspace.decouple_cell(cl_th_cpl)

    cl_th_dec_jl = PyCall.convert(Array{Float64,2}, cl_th_dec)
    cl_EE_th = cl_th_dec_jl[1, :]
    cl_BB_th = cl_th_dec_jl[4, :]

    return cl_EE_th, cl_BB_th, plan.leff
end
