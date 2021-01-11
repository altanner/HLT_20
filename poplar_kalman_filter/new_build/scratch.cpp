#include <iostream>

void count()
{
    for (uint i = 0; i < 4; i++)
        {
            std::cout << i << "\n";
        }
}

int bob;

int a_number()
{
    int bob = 1342;
    return bob;
}

int main()
{
    bob = a_number();
    std::cout << bob << "\n";
}