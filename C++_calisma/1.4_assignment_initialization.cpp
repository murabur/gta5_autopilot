#include <iostream>

int main()
{
    int x;  //değişken tanımlama - define
    x = 10; //değişken atama - assignment

    std::cout << "x = 15 demeden once : " << x << "\n"; 

    x = 15;

    std::cout << "x = 15 dedikten sonra : " << x << "\n"; 

    // değişken başlatma - Variable initialization

    int y { 20 }; // y değişkenini tanımlayıp 20 değerini atar
    std::cout << "y degiskeninin degeri : " << y ;


    //------------------------------------------------------

    /* Initialization (İlklendirme)

    Bir değişken oluşturulduğu anda ona ilk değerinin verilmesidir. 
    Bellekte yer ayrılır ve o an değer içine yazılır.

    Önem: const ve referanslar( & ) sadece initialization yapılabilir, 
    sonradan atama yapılamaz.

    Daha performanslıdır. Bellekte çöp değer kalma riskini sıfırlar.

    */

    int a { 10 }; // Initialization (Modern ve Güvenli)
    int b;        // Sadece tanımlama (Tehlikeli, çöp değer)
    b = 20;       // Assignment (Sonradan değer verme)


    return 0;
}