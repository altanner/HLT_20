#include <iostream>
#include <exception>
#include <vector>
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
using poplar::DataStream;
using poplar::program::Sequence;
using poplar::program::Copy;


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


std::vector<Tensor> mapInputsToIPU(Graph &graph, int n_inputs, uint batch_size)
{
    std::vector<Tensor> inputs(n_inputs);
    std::vector<Tensor> inputs_batch(n_inputs);

    for (int i = 0; i < inputs.size(); i++)
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

    return inputs_batch;
}


void mapCodeletsToVertices(Graph &graph)
{
    popops::addCodelets(graph);
    poplin::addCodelets(graph);
    graph.addCodelets("matrixInverseVertex.cpp");
    graph.addCodelets("matrixProductVertex.cpp");
    graph.addCodelets("scaledAddVertex.cpp");
    graph.addCodelets("packHitsVertex.cpp");

    std::cout << "Codelets mapped to graph vertices." << "\n";
}

void mapInStreamsToDevice(Graph &graph,
                          uint batch_size,
                          std::vector<Tensor> inputs_batch,
                          std::vector<Tensor> covs,
                          std::vector<DataStream> inStreams,
                          Sequence preProg)
{
    for (uint i = 0; i < covs.size(); i++)
        {
            std::string iStr = std::to_string(i);
            inStreams[i] = graph.addHostToDeviceFIFO("inStream" + iStr,
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
    int n_inputs = 1;
    unsigned int batch_size = 1;

    //~ 53
    Device dev = connectToIPU();

    //~ 56
    Graph graph(dev);

    //~ 61-72
    std::vector<Tensor> inputs_batch;
    inputs_batch = mapInputsToIPU(graph, n_inputs, batch_size);

    //~ 75-80
    mapCodeletsToVertices(graph);

    //// std::vector<DataStream> inStreams(n_inputs);
    //// std::vector<Tensor> covs(n_inputs);
    // std::vector<Tensor> qs(n_inputs);
    // std::vector<Tensor> hs(n_inputs);
    // std::vector<Tensor> gs(n_inputs);
    // std::vector<Tensor> fs(n_inputs);
    // std::vector<Tensor> d(n_inputs);
    // std::vector<Tensor> dInit(n_inputs);
    // std::vector<Tensor> dSkip(n_inputs);
    // std::vector<Tensor> scatterInto(n_inputs);
    // std::vector<Tensor> loop(n_inputs);
    // std::vector<Tensor> zero(n_inputs);
    // std::vector<Tensor> one(n_inputs);
    // std::vector<Tensor> loop_batch(n_inputs);
    // std::vector<Tensor> hitThisLoop(n_inputs);
    // //~ projection tensors (n_inputs)
    // std::vector<Tensor> p_proj_all(n_inputs);
    // std::vector<Tensor> C_proj_all(n_inputs);
    // //~ kalman filter tensors (n_inputs)
    // std::vector<Tensor> p_filt_all(n_inputs);
    // std::vector<Tensor> C_filt_all(n_inputs);
    // //~ backward smoothing tensors (n_inputs)
    // std::vector<Tensor> p_smooth(n_inputs);
    // std::vector<Tensor> C_smooth(n_inputs);
    //~ flattened covariance tensor  //? what is a flat cov?
    // std::vector<Tensor> covFlat(covs.size());

    std::vector<Tensor> covs(n_inputs);
    std::vector<DataStream> inStreams(n_inputs);
    //~ 111-117
    Sequence preProg;
    mapInStreamsToDevice(graph,
                         batch_size,
                         inputs_batch,
                         covs,
                         inStreams,
                         preProg);


    //Sequence prog;


    return 0;
}
