#include <iostream>
#include <random>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <fstream>
using namespace xt::placeholders; // to enable _ syntax
using namespace std;
using namespace xt;

void gnuplot(xarray<double> two_dimensional_dataset, xarray<int> true_pred)
{

    // Define paths
    string gnuplot_commands_path = "temp/gnuplot_commands.gp";
    string data_path = "temp/gnuplot.dat";

    // Retrieve dataset size
    int n_rows = two_dimensional_dataset.shape()[0];

    // Create a data file
    std::ofstream data(data_path);
    for (int i = 0; i < n_rows; i++)
    {
        data
            << two_dimensional_dataset(i, 0) << " "
            << two_dimensional_dataset(i, 1) << " "
            << true_pred(i, 0)
            << endl; // x and y values
    }
    data.close();

    // Use Gnuplot to plot the data
    std::ofstream gnuplot_commands(gnuplot_commands_path);
    gnuplot_commands << "set terminal wxt size 1200,800\n";         // Set window size
    gnuplot_commands << "set xlabel 'x1'\n";                        // Set x-axis label
    gnuplot_commands << "set ylabel 'x2'\n";                        // Set y-axis label
    gnuplot_commands << "set title 'x1 vs x2 Colored by Cluster y'\n"; // Set plot title
    gnuplot_commands << "set pointsize 0.5\n";                        // Set point size
    gnuplot_commands << "set palette defined (0 'blue', 1 'red')\n"; // Define color palette for clusters
    gnuplot_commands << "plot 'temp/gnuplot.dat' u 1:2:3 with points palette pt 7\n"; // Plot data points using the third column for color

    // Close the file
    gnuplot_commands.close();

    // Run gnuplot with the command file
    std::system(("unset GTK_PATH && gnuplot-x11 -persist " + gnuplot_commands_path).c_str());
}

    xt::xarray<double> create_random_dataset(float mean, float variance, int n_observations)
    {
        // Define seed
        std::random_device rd;
        std::mt19937 e2(rd());

        std::normal_distribution<> dist(mean, variance);

        // Init data array
        xt::xarray<double> dataset = xt::zeros<double>({0, 2});
        for (int n = 0; n < n_observations; n++)
        {
            xt::xarray<double> data_point = {dist(e2), dist(e2)};
            data_point.reshape({1, 2});
            dataset = xt::concatenate(xtuple(dataset, data_point), 0);
        }

        return dataset;
    }

    void shuffleArray(xt::xarray<double> & array)
    {
        // Get the number of rows
        std::size_t rows = array.shape(0);

        // Create an index array to shuffle
        xt::xarray<std::size_t> indices = xt::arange<std::size_t>(rows);

        // Shuffle the indices
        xt::random::shuffle(indices);

        // Create a temporary array to hold the shuffled rows
        xt::xarray<double> temp = xt::empty_like(array);

        // Fill the temporary array with shuffled rows
        for (std::size_t i = 0; i < rows; ++i)
        {
            temp(i, 0) = array(indices(i), 0);
            temp(i, 1) = array(indices(i), 1);
            temp(i, 2) = array(indices(i), 2);
        }

        // Copy the shuffled rows back to the original array
        array = temp;
    }

    xarray<double> sigma(xarray<double> x)
    {
        return 1 / (1 + xt::exp(-x));
    }

    xarray<double> sigma_derivative(xarray<double> x)
    {
        return sigma(x) * (1 - sigma(x));
    }
