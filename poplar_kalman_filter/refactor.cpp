//? what is the class concerned with?
//? how do I turn loops into functions?
//? what will it return? nothing?
//? what args will it take?

void map_to_tiles(inputs, inputs_batch, graph)
{
    for (uint i = 0; i < inputs.size(); i++)
        {
            std::string iStr = std::to_string(i);
            inputs[i] = graph.addVariable(FLOAT, {5, 2}, "x_in" + iStr);
            inputs_batch[i] = graph.addVariable(FLOAT, {uint(batch_size), 5 * 2}, "x_in_batch" + iStr); // Check dims!
            graph.setTileMapping(inputs[i], i);
            graph.setTileMapping(inputs_batch[i], i);
        }
}