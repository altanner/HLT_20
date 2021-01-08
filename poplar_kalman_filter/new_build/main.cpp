#include <iostream>
#include <exception>
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
//using poplar::Vertex;


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

//void codeletSetUp():


int main()
{
    Device dev = connectToIPU();

    Graph graph(dev);

    int n_inputs = 1;
    unsigned int batch_size = 1;
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

    popops::addCodelets(graph);
    poplin::addCodelets(graph);
    graph.addCodelets("matrixInverseVertex.cpp");
    graph.addCodelets("matrixProductVertex.cpp");
    graph.addCodelets("scaledAddVertex.cpp");
    graph.addCodelets("packHitsVertex.cpp");

    return 0;
}
