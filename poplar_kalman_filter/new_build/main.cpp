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

using poplar::DeviceManager;
using poplar::Device;
using poplar::Graph;
using poplar::Tensor;
using poplar::FLOAT;
using poplar::INT;
using poplar::DataStream;
using poplar::program::Sequence;
using poplar::program::Copy;

int N = 5;               // Number of planes
float d = 1.0;           // Distance between planes
float sigma = 10E-2;     // Resolution of planes
float z = 0.1;           // Thickness of absorber
float x0 = 0.01;         // Radiation length of absorber
float theta0 = 10E-3;    // Multiple scatter uncertainty


Device connectToIPU()
{
    bool success = false;
    DeviceManager manager = DeviceManager::createDeviceManager();
    Device dev;
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

void mapInputsToTiles(Graph &graph,
                    unsigned int n_inputs,
                    unsigned int batch_size,
                    std::vector<Tensor> &inputs,
                    std::vector<Tensor> &inputs_batch)
{
    inputs.resize(n_inputs);       //~ this reduces the input dependencies
    inputs_batch.resize(n_inputs); //~ to being JUST n_inputs

    for (int i = 0; i < n_inputs; i++)
    {
        inputs[i] = graph.addVariable(FLOAT,
                                      {5, 2},
                                      "x_in" + std::to_string(i));
        inputs_batch[i] = graph.addVariable(FLOAT,
                                            {batch_size, 5 * 2},
                                            "x_in_batch" + std::to_string(i));
        graph.setTileMapping(inputs[i], i);
        graph.setTileMapping(inputs_batch[i], i);
    }

    std::cout << n_inputs << " inputs of "
              << batch_size << " batches mapped to tiles." << "\n";
}

void addProgramCodelets(Graph &graph)
{
    popops::addCodelets(graph);
    poplin::addCodelets(graph);
    graph.addCodelets("matrixInverseVertex.cpp");
    graph.addCodelets("matrixProductVertex.cpp");
    graph.addCodelets("scaledAddVertex.cpp");
    graph.addCodelets("packHitsVertex.cpp");

    std::cout << "Codelets added." << "\n";
}

void mapInStreamsToDevice(Graph &graph,
                          unsigned int batch_size,
                          std::vector<Tensor> &inputs_batch,
                          std::vector<DataStream> &inStreams,
                          Sequence &preProg)
{
    unsigned int n_inputs = inputs_batch.size();
    inStreams.resize(n_inputs);
    for (uint i = 0; i < n_inputs; i++)
        {
            inStreams[i] = graph.addHostToDeviceFIFO("inStream" + std::to_string(i),
                                                    FLOAT,
                                                    5 * 2 * batch_size);
            preProg.add(Copy(inStreams[i], inputs_batch[i]));
            std::cout << "streamloop" << "\n";
        }
}

//~//~//~//~//~//~//~//~//~//~//~//~//~//~//~//~//~//~//~//~//~

int main()
{
    //~ 59-60
    unsigned int n_inputs = 1;
    unsigned int batch_size = 1;

    //~ 53
    Device dev = connectToIPU();

    //~ 61-72
    Graph graph(dev);
    std::vector<Tensor> inputs, inputs_batch;
    mapInputsToTiles(graph,
                     n_inputs,
                     batch_size,
                     inputs,
                     inputs_batch);

    //~ 75-80
    addProgramCodelets(graph);

    //§ preProg §//§//§//§//§//§//§//§//§//§//§//§
    std::vector<Tensor> covs;
    std::vector<DataStream> inStreams;
    //~ 111-117
    Sequence preProg;
    mapInStreamsToDevice(graph,
                         batch_size,
                         inputs_batch,
                         inStreams,
                         preProg);

    //§ prog §//§//§//§//§//§//§//§//§//§//§//§//§
    //~ I tried to make prog into a func, but seeing as it has
    // //~ 20 args going in, I guess that is naive.
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

//     //~ 120-217
    Sequence prog;
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

        //! this line > segfault
        prog.add(Copy(inputs_batch[i].slice(0, 1, 0).reshape({5, 2}), inputs[i]));

//         //* line 25
//         //? why are these inside the loop? tf they are out(?)
//         //~ initiate the matrices as tensors
//         //~ F is the transfer matrix
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

   } //~ end for (uint i = 0; i < covs.size(); i++)

    return 0;
}
