
TYPE initGraph()
{
    //~ instantiate graph
    Graph graph(dev.getTarget());

    popops::addCodelets(graph);
    poplin::addCodelets(graph);
    graph.addCodelets("matrixInverseVertex.cpp");
    graph.addCodelets("matrixProductVertex.cpp");
    graph.addCodelets("scaledAddVertex.cpp");
    graph.addCodelets("packHitsVertex.cpp");
}

