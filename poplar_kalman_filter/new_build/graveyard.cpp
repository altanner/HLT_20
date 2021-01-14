void progLoop(Graph &graph,
              std::vector<Tensor> qs,
              std::vector<Tensor> hs,
              std::vector<Tensor> gs,
              std::vector<Tensor> fs,
              std::vector<Tensor> d,
              std::vector<Tensor> dInit,
              std::vector<Tensor> dSkip,
              std::vector<Tensor> scatterInto,
              std::vector<Tensor> loop,
              std::vector<Tensor> zero,
              std::vector<Tensor> one,
              std::vector<Tensor> loop_batch,
              std::vector<Tensor> hitThisLoop,
              //~ projection tensors (n_inputs)
              std::vector<Tensor> p_proj_all,
              std::vector<Tensor> C_proj_all,
              //~ kalman filter tensors (n_inputs)
              std::vector<Tensor> p_filt_all,
              std::vector<Tensor> C_filt_all,
              //~ backward smoothing tensors (n_inputs)
              std::vector<Tensor> p_smooth,
              std::vector<Tensor> C_smooth,
              std::vector<Tensor> covs,
              std::vector<Tensor> covFlat)
{
    for (uint i = 0; i < covs.size(); i++)
    {
        std::string iStr = std::to_string(i);
        loop[i] = graph.addVariable(INT, {1}, "loop");
        graph.setTileMapping(loop[i], i);

        scatterInto[i] = graph.addConstant<int>(INT, {1}, {0});
        graph.setTileMapping(scatterInto[i], i);

        zero[i] = graph.addConstant<int>(INT, {1}, {0});
        graph.setTileMapping(zero[i], i);

        one[i] = graph.addConstant<int>(INT, {1}, {1});
        graph.setTileMapping(one[i], i);

        prog.add(Copy(inputs_batch[i].slice(0, 1, 0).reshape({5, 2}), inputs[i]));

        //* line 25
        //? why are these inside the loop? tf they are out(?)
        //~ initiate the matrices as tensors
        //~ F is the transfer matrix
        Tensor fFlat = graph.addConstant<float>(FLOAT, {16, 1},
                                                {1., 1., 0., 0.,
                                                0., 1., 0., 0.,
                                                0., 0., 1., 1.,
                                                0., 0., 0., 1.});

        //~ G is the noise matrix
        Tensor gFlat = graph.addConstant<float>(FLOAT, {16, 1},
                                                {float(1.0)/(sigma * sigma), 0., 0., 0.,
                                                0, 0., 0., 0.,
                                                0, 0., float(1.0)/(sigma * sigma), 0.,
                                                0, 0., 0., 0.});

        //~ H the relation between the measurement m and the state p
        Tensor hFlat = graph.addConstant<float>(FLOAT, {16, 1},
                                                {1., 0., 0., 0.,
                                                0., 0., 0., 0.,
                                                0., 0., 1., 0.,
                                                0., 0., 0., 0.});

        //~ Q is the random error matrix, ie the scatter
        Tensor qFlat = graph.addConstant<float>(FLOAT, {16, 1}, {0.});

        //~ cov is the initial parameters
        covFlat[i] = graph.addConstant<float>(FLOAT, {16, 1},
                                              {sigma * sigma, 0., 0., 0.,
                                              0., M_PI, 0., 0.,
                                              0., 0., sigma * sigma, 0.,
                                              0., 0., 0., M_PI});

        d[i] = graph.addVariable(FLOAT, {1, 1}, "d" + iStr);
        dInit[i] = graph.addConstant<float>(FLOAT, {1, 1}, {1.});
        dSkip[i] = graph.addConstant<float>(FLOAT, {1, 1}, {2.});

        prog.add(Copy(dInit[i], d[i]));

        covs[i] = graph.addVariable(FLOAT, {4, 4}, "cov" + iStr);
        prog.add(Copy(covFlat[i].reshape({4, 4}), covs[i]));

        p_proj_all[i] = graph.addVariable(FLOAT, {5, 4, 1}, "p_proj_all" + iStr);
        C_proj_all[i] = graph.addVariable(FLOAT, {5, 4, 4}, "C_proj_all" + iStr);

        p_filt_all[i] = graph.addVariable(FLOAT, {5, 4, 1}, "p_filt_all" + iStr);
        C_filt_all[i] = graph.addVariable(FLOAT, {5, 4, 4}, "C_filt_all" + iStr);

        p_smooth[i] = graph.addVariable(FLOAT, {4, 1}, "p_smooth" + iStr);
        C_smooth[i] = graph.addVariable(FLOAT, {4, 4}, "C_smooth" + iStr);

        graph.setTileMapping(p_proj_all[i], i);
        graph.setTileMapping(C_proj_all[i], i);
        graph.setTileMapping(p_filt_all[i], i);
        graph.setTileMapping(C_filt_all[i], i);

        graph.setTileMapping(p_smooth[i], i);
        graph.setTileMapping(C_smooth[i], i);

        qs[i] = qFlat.reshape({4, 4});
        hs[i] = hFlat.reshape({4, 4});
        gs[i] = gFlat.reshape({4, 4});
        fs[i] = fFlat.reshape({4, 4});

        graph.setTileMapping(covFlat[i], i);
        graph.setTileMapping(qFlat, i);
        graph.setTileMapping(d[i], i);
        graph.setTileMapping(dInit[i], i);
        graph.setTileMapping(dSkip[i], i);
        graph.setTileMapping(covs[i], i);
        graph.setTileMapping(qs[i], i);
        graph.setTileMapping(gFlat, i);
        graph.setTileMapping(hFlat, i);
        graph.setTileMapping(fFlat, i);
        graph.setTileMapping(hs[i], i);
        graph.setTileMapping(gs[i], i);
        graph.setTileMapping(fs[i], i);

    }
}


//~ I tried to make prog into a func, but seeing as it has
//~ 20 args going in, I guess that is naive.
    std::vector<Tensor> qs(n_inputs);
    std::vector<Tensor> hs(n_inputs);
    std::vector<Tensor> gs(n_inputs);
    std::vector<Tensor> fs(n_inputs);
    std::vector<Tensor> d(n_inputs);
    std::vector<Tensor> dInit(n_inputs);
    std::vector<Tensor> dSkip(n_inputs);
    std::vector<Tensor> scatterInto(n_inputs);
    std::vector<Tensor> loop(n_inputs);
    std::vector<Tensor> zero(n_inputs);
    std::vector<Tensor> one(n_inputs);
    std::vector<Tensor> loop_batch(n_inputs);
    std::vector<Tensor> hitThisLoop(n_inputs);
    //~ projection tensors (n_inputs)
    std::vector<Tensor> p_proj_all(n_inputs);
    std::vector<Tensor> C_proj_all(n_inputs);
    //~ kalman filter tensors (n_inputs)
    std::vector<Tensor> p_filt_all(n_inputs);
    std::vector<Tensor> C_filt_all(n_inputs);
    //~ backward smoothing tensors (n_inputs)
    std::vector<Tensor> p_smooth(n_inputs);
    std::vector<Tensor> C_smooth(n_inputs);
    std::vector<Tensor> covFlat(covs.size());

    //~ 120-217
    std::vector<Tensor> inputs(n_inputs);
//    std::vector<Tensor> inputs_batch(n_inputs);