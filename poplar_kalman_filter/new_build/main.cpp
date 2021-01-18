#include <iostream>
#include <exception>
#include <vector>
#include <math.h>
#include <unistd.h> //~ UNIX standard w glibc

#include <poplar/DeviceManager.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

#include <popops/codelets.hpp>

#include <poplin/codelets.hpp>

//~ the -using- lines have moved to graveyard
//~ resurrect once you are clear on what is being abbreviated

int N = 5;               // Number of planes
float d = 1.0;           // Distance between planes
float sigma = 10E-2;     // Resolution of planes
float z = 0.1;           // Thickness of absorber
float x0 = 0.01;         // Radiation length of absorber
float theta0 = 10E-3;    // Multiple scatter uncertainty

poplar::Device connectToIPU()
{
    bool success = false;
    poplar::DeviceManager manager = poplar::DeviceManager::createDeviceManager();
    poplar::Device dev;
    auto devices = manager.getDevices(poplar::TargetType::IPU, 1);

    for (auto &device : devices)
    {
        dev = std::move(device);
        std::cout << "Trying to attach";
        success = dev.attach();
        if (success)
        {
            std::cout << " - ok, attached to IPU " << dev.getId() << "\n";
            break;
        }
        else
        {
            std::cout << " - hmm, issue attaching to IPU " << dev.getId() << "\n";
        }
    }

    if (!success)
    {
        throw std::runtime_error("Cannot connect to IPU.");
    }

    return dev;
}

void mapInputsToTiles(poplar::Graph &graph,
                      unsigned int n_inputs,
                      unsigned int batch_size,
                      std::vector<poplar::Tensor> &inputs,
                      std::vector<poplar::Tensor> &inputs_batch)
{
    inputs.resize(n_inputs);       //~ this reduces the input dependencies
    inputs_batch.resize(n_inputs); //~ to being JUST n_inputs

    for (int i = 0; i < n_inputs; i++)
    {                             //? why do strings go in to these tensors?
        inputs[i] = graph.addVariable(poplar::FLOAT,
                                      {5, 2},
                                      "x_in" + std::to_string(i));
        inputs_batch[i] = graph.addVariable(poplar::FLOAT,
                                            {batch_size, 5 * 2},
                                            "x_in_batch" + std::to_string(i));
        graph.setTileMapping(inputs[i], i);
        graph.setTileMapping(inputs_batch[i], i);
    }

    std::cout << n_inputs << " inputs of "
              << batch_size << " batches mapped to tiles." << "\n";
}

void addProgramCodelets(poplar::Graph &graph)
{
    popops::addCodelets(graph);
    poplin::addCodelets(graph);
    graph.addCodelets("matrixInverseVertex.cpp");
    graph.addCodelets("matrixProductVertex.cpp");
    graph.addCodelets("scaledAddVertex.cpp");
    graph.addCodelets("packHitsVertex.cpp");

    std::cout << "Codelets added." << "\n";
}

void sendInStreamsToDevice(poplar::Graph &graph,
                          unsigned int batch_size,
                          std::vector<poplar::Tensor> &inputs_batch,
                          std::vector<poplar::DataStream> &inStreams,
                          poplar::program::Sequence &preProg)
{
    unsigned int n_inputs = inputs_batch.size();
    inStreams.resize(n_inputs);
    for (unsigned int i = 0; i < n_inputs; i++)
        {
            inStreams[i] = graph.addHostToDeviceFIFO("inStream" + std::to_string(i),
                                                    poplar::FLOAT,
                                                    5 * 2 * batch_size);
            preProg.add(poplar::program::Copy(inStreams[i], inputs_batch[i]));
        }
    std::cout << "Datastreams sent to IPU." << "\n";
}

//~//~//~//~//~//~//~//~//~//~//~//~//~//~//~//~//~//~//~//~//~

int main()
{

    //~ 53
    poplar::Device dev = connectToIPU();

    //~ 61-72
    //~ 59-60
    poplar::Graph graph(dev);
    unsigned int n_inputs = 1;
    unsigned int batch_size = 1;
    std::vector<poplar::Tensor> inputs,
                                inputs_batch;
    mapInputsToTiles(graph,
                     n_inputs,
                     batch_size,
                     inputs,
                     inputs_batch);

    //~ 75-80
    addProgramCodelets(graph);

    //§ preProg §//§//§//§//§//§//§//§//§//§//§//§//§
    //~ 111-117
    std::vector<poplar::DataStream> inStreams(n_inputs);
    poplar::program::Sequence preProg;
    sendInStreamsToDevice(graph,
                          batch_size,
                          inputs_batch,
                          inStreams,
                          preProg);

    //§ prog declarations §//§//§//§//§//§//§//§//§//§
    std::vector<poplar::Tensor> qs(n_inputs),
                                hs(n_inputs),
                                gs(n_inputs),
                                fs(n_inputs),
                                d(n_inputs),
                                dInit(n_inputs),
                                dSkip(n_inputs),
                                scatterInto(n_inputs),
                                loop(n_inputs),
                                zero(n_inputs),
                                one(n_inputs),
                                loop_batch(n_inputs),
                                hitThisLoop(n_inputs),
                                //~ projection tensors
                                p_proj_all(n_inputs),
                                C_proj_all(n_inputs),
                                //~ kalman filter tensors
                                p_filt_all(n_inputs),
                                C_filt_all(n_inputs),
                                //~ backward smoothing tensors
                                p_smooth(n_inputs),
                                C_smooth(n_inputs),
                                //~ flattened covariance tensor
                                covs(n_inputs),
                                covsFlat(n_inputs);

    //~ these xFlat tensors used to be declared in loop
    //~ F is the transfer matrix
    poplar::Tensor fFlat =
        graph.addConstant<float>(poplar::FLOAT, {16, 1},
                                {1., 1., 0., 0.,
                                 0., 1., 0., 0.,
                                 0., 0., 1., 1.,
                                 0., 0., 0., 1.});

    //~ G is the noise matrix
    poplar::Tensor gFlat =
        graph.addConstant<float>(poplar::FLOAT, {16, 1},
                                {float(1.0)/(sigma * sigma), 0., 0., 0.,
                                 0, 0., 0., 0.,
                                 0, 0., float(1.0)/(sigma * sigma), 0.,
                                 0, 0., 0., 0.});

    //~ H the relation between the measurement m and the state p
    poplar::Tensor hFlat =
        graph.addConstant<float>(poplar::FLOAT, {16, 1},
                                {1., 0., 0., 0.,
                                 0., 0., 0., 0.,
                                 0., 0., 1., 0.,
                                 0., 0., 0., 0.});

    //~ Q is the random error matrix, ie the scatter (?why)
    poplar::Tensor qFlat =
        graph.addConstant<float>(poplar::FLOAT,
                                {16, 1},
                                {0.});

    //~ 120-217
    //§ prog §//§//§//§//§//§//§//§//§//§//§//§//§
    poplar::program::Sequence prog;
    for (unsigned int i = 0; i < n_inputs; i++)
    {
        std::string iStr = std::to_string(i);
        //~ what plane of KF I think
        //? why do strings go into these tensors?
        loop[i] = graph.addVariable(poplar::INT, {1}, "loop");
        graph.setTileMapping(loop[i], i);

        //~ btw nothing to do with particle scattering
        //? why do we have <int> and poplar::INT?
        scatterInto[i] = graph.addConstant<int>(poplar::INT, {1}, {0});
        graph.setTileMapping(scatterInto[i], i);

        //? a tensor of zeros?
        zero[i] = graph.addConstant<int>(poplar::INT, {1}, {0});
        graph.setTileMapping(zero[i], i);

        one[i] = graph.addConstant<int>(poplar::INT, {1}, {1});
        graph.setTileMapping(one[i], i);

        prog.add(poplar::program::Copy(inputs_batch[i].slice(0, 1, 0).reshape({5, 2}),
                      inputs[i]));

        d[i] =
            graph.addVariable(poplar::FLOAT, {1, 1}, "d" + iStr);
        dInit[i] =
            graph.addConstant<float>(poplar::FLOAT, {1, 1}, {1.});
        dSkip[i] =
            graph.addConstant<float>(poplar::FLOAT, {1, 1}, {2.});

        prog.add(poplar::program::Copy(dInit[i], d[i]));

        covs[i] =
            graph.addVariable(poplar::FLOAT, {4, 4}, "cov" + iStr);

        //~ cov is the initial parameters (?)
        covsFlat[i] =
            graph.addConstant<float>(poplar::FLOAT, {16, 1},
                                    {sigma * sigma, 0., 0., 0.,
                                     0., M_PI, 0., 0.,
                                     0., 0., sigma * sigma, 0.,
                                     0., 0., 0., M_PI});

        //~ prog.adds can go to function?
        prog.add(poplar::program::Copy(covsFlat[i].reshape({4, 4}), covs[i]));

        //~ graph add are just building the graph (to func?/class)
        p_proj_all[i] = graph.addVariable(poplar::FLOAT,
                                         {5, 4, 1},
                                         "p_proj_all" + iStr);

        C_proj_all[i] = graph.addVariable(poplar::FLOAT,
                                         {5, 4, 4},
                                         "C_proj_all" + iStr);

        p_filt_all[i] = graph.addVariable(poplar::FLOAT,
                                         {5, 4, 1},
                                         "p_filt_all" + iStr);

        C_filt_all[i] = graph.addVariable(poplar::FLOAT,
                                         {5, 4, 4},
                                         "C_filt_all" + iStr);

        p_smooth[i] = graph.addVariable(poplar::FLOAT,
                                       {4, 1},
                                       "p_smooth" + iStr);

        C_smooth[i] = graph.addVariable(poplar::FLOAT,
                                       {4, 4},
                                       "C_smooth" + iStr);

        graph.setTileMapping(p_proj_all[i], i);
        graph.setTileMapping(C_proj_all[i], i);
        graph.setTileMapping(p_filt_all[i], i);
        graph.setTileMapping(C_filt_all[i], i);

        graph.setTileMapping(p_smooth[i], i);
        graph.setTileMapping(C_smooth[i], i);

        graph.setTileMapping(covs[i], i);
        graph.setTileMapping(covsFlat[i], i);

        graph.setTileMapping(d[i], i);
        graph.setTileMapping(dInit[i], i);
        graph.setTileMapping(dSkip[i], i);

        fs[i] = fFlat.reshape({4, 4});
        graph.setTileMapping(fs[i], i);
        graph.setTileMapping(fFlat, i);
        gs[i] = gFlat.reshape({4, 4});
        graph.setTileMapping(gs[i], i);
        graph.setTileMapping(gFlat, i);
        hs[i] = hFlat.reshape({4, 4});
        graph.setTileMapping(hs[i], i);
        graph.setTileMapping(hFlat, i);
        qs[i] = qFlat.reshape({4, 4});
        graph.setTileMapping(qs[i], i);
        graph.setTileMapping(qFlat, i);

    } //~ end for (unsigned int i = 0; i < n_inputs; i++)

    return 0;
}

