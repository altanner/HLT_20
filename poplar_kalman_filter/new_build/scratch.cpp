// #include <iostream>

// void count()
// {
//     for (uint i = 0; i < 4; i++)
//         {
//             std::cout << i << "\n";
//         }
// }

// int bob;

// int a_number()
// {
//     int bob = 1342;
//     return bob;
// }

// int main()
// {
//     bob = a_number();
//     std::cout << bob << "\n";
// }


#include <iostream>

using namespace std;

int main()
{
    long int x;            // A normal integer
    long int *p;           // A pointer to an integer

    p = &x;           // Read it, "assign the address of x to p"
    cin >> x;          // Put a value in x, we could also use *p here
    cin.ignore();
    cout << *p << "\n"; // Note the use of the * to get the value
    cin.get();
}