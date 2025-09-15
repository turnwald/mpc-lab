def test_imports():
    import mpc_lab
    import mpc_lab.core.types as _t
    import mpc_lab.core.specs as _s
    import mpc_lab.core.traj as _traj
    import mpc_lab.modeling.dynamics.diffdrive as _dd
    import mpc_lab.modeling.dynamics.attitude_quat as _aq
    import mpc_lab.transcription.qp_builder as _qpb
    import mpc_lab.transcription.nlp_builder as _nlp
    import mpc_lab.solvers.qp_facade as _qpf
    import mpc_lab.solvers.nlp_facade as _nlpf
    import mpc_lab.controllers.base as _cb
    import mpc_lab.controllers.mpc_linear as _ml
    import mpc_lab.controllers.mpc_koopman as _mk
    import mpc_lab.controllers.mpc_nonlinear as _mn
    import mpc_lab.controllers.cbf_filter as _cf
    import mpc_lab.koopman.features as _kf
    import mpc_lab.safety.cbf as _scbf
    # If we got here, the package skeleton imports cleanly.
