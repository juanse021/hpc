#include<bits/stdc++.h>

using namespace std;
const int MX = 5000;

int main () {
  cout << MX << endl << MX << endl;
  for (int i = 0; i < MX; i++) {
    for (int j = 0; j < MX; j++) {
      cout << i*2.7 + j*0.4;
      if (j + 1 < MX) cout << ", ";
    }
    cout << endl;
  }
}
