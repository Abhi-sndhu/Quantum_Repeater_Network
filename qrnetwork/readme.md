<!-- Title Page -->

<div align="center">

# 🛰️ **QNeCT**
## *Quantum Network and Communication Toolbox*
</div>

## Table of Contents
1. [About](#about)
2. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
3. [Usage](#usage)
    - [Fundamental operations on quantum system](#qobj)
4. [Contributing](#contributing)
5. [License](#license)
6. [Contact](#contact)
7. [Acknowledgements](#acknowledgements)

## About

**QNeCT** is an open-source software library designed primarily for simulating **quantum repeater–based networks**. In addition, it provides tools to model and analyze **complex quantum mechanical systems**. It is built upon well-established Python packages such as [**NumPy**](https://numpy.org/), [**SciPy**](https://scipy.org/), and [**SymPy**](https://www.sympy.org/en/index.html). QNeCT offers a flexible and extensible framework for research and experimentation. It is freely available for use and modification on all major platforms, including **Linux**, **macOS**, and **Windows**.


<!-- ![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-1.0.0-green.svg) -->





**Key Features:**
* Solve complex quantum mechanical problems with ease.  
* Analyze the characteristics and performance of quantum channels.  
* Evaluate the nature and quality of entanglement distributed over a quantum repeater–based networks.
 

<!-- ## Built With
List the main technologies, frameworks, and libraries used.
*   [Technology](https://link-to-technology.com)
*   [Framework](https://link-to-framework.com)
*   [Library](https://link-to-library.com) -->

## Getting Started

### Prerequisites
You must have pre installed python environments such as [**Anaconda**](https://www.anaconda.com/download) and [**VS Code**](https://code.visualstudio.com/download) to use the **QNeCT** packages.

### Installation

A step-by-step guide to install and set up the project.

1.  Clone the repo
    ```sh
    https://github.com/QNeCT-India
    ```
2.  Navigate to the project directory
    ```sh
    cd your_repo_name
3.  Install required packages
    ```sh
    npm install
4.  (Optional) Set up environment variables in a `.env` file
    ```env
    API_KEY = 'ENTER YOUR API';

## Functions Overview

A comprehensive instructions are provided to assist users in utilizing the existing functions of **QNeCT**.

### Fundamental operations on quantum system
Quantum states are represented either by a complex wave function or a complex column vector. In **QNeCT**, operations that are frequently used in quantum mechanical systems such as dagger (complex conjugate transpose), normalization, and construction of density matrix states are implemented in the **Q** class. Additionally, commonly used quantum states (e.g., Bell states) and operations (e.g., Pauli operators) are also provided for convenience.

| Functions | Usage |
|--|--|
| `Q(input).d()` | Gives dagger of an input matrix array |
| `Q(input).n()` | Gives norm of an input matrix array |
| `Q(input).u()` | Gives normalized state of an input matrix array |
| `Q(input).dm()` | Gives density matrix if input array is a ket vector |
| `Q(input).purity()` | Gives purity as 1 if an input matrix array is a ket vector and calculates `tr(input^2)` if input array is a square matrix |
| `Q.rand.Haar(n)` | Gives a random Haar unitary matrix of size n × n |
| `Q.rand.p(n)` | Gives a random n-qubit pure state |
| `Q.rand.m(n)` | Gives a random n-qubit mixed state |
| `Q.Bell.phi_plus, /phi_minus/psi_plus/psi_minus` | Gives one of the four Bell states |
| `Q.Bell.all()` | Gives all the four Bell states |
| `Q.Pauli.I/X/Y/Z` | Gives one of the Pauli Matrices |
| `Q.Pauli.all()` | Gives all the Pauli matrices |


<!-- Run the development server:
```sh
npm start
``` -->

## Contributing

<!-- Guidelines for others who want to contribute to your project.

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Your Name - [@your_twitter](https://twitter.com/your_twitter) - email@example.com

Project Link: [https://github.com/your_username/your_repo_name](https://github.com/your_username/your_repo_name)

## Acknowledgements

Credit to any resources, tutorials, or people that helped.

*   [Shields.io](https://shields.io/)
*   [Choose an Open Source License](https://choosealicense.com)
*   [GitHub Emoji Cheat Sheet](https://www.webfx.com/tools/emoji-cheat-sheet/)