#pragma once

#include <memory>
#include <vector>
#include <numeric>
#include <functional>


template <class T>
class Tensor
{
public:
    Tensor() = default;
    Tensor(const std::vector<int> &dimSizes) :
        _dimSizes(dimSizes),
        _actuallyDimSizes(dimSizes)
    {
        const int allDim = std::accumulate(_dimSizes.cbegin(), _dimSizes.cend(), 1, std::multiplies<int>());
        _data = std::make_shared<std::vector<T>>(allDim);
    }
    ~Tensor() = default;


    Tensor(const Tensor & rhs):
        _dimSizes(rhs._dimSizes),
        _actuallyDimSizes(rhs._actuallyDimSizes),
        _modifications(rhs._modifications)
    {
        _data = std::make_shared<std::vector<T>>(*rhs._data);
    }

    Tensor & operator=(const Tensor &rhs)
    {
        if (this != &rhs)
        {
            _dimSizes = rhs._dimSizes;
            _actuallyDimSizes = rhs._actuallyDimSizes;
            _data = std::make_shared<std::vector<T>>(*rhs._data);
            _modifications = rhs._modifications;
        }

        return *this;
    }

    Tensor(Tensor &&rhs) = default;
    Tensor & operator=(Tensor &&rhs) = default;

    T& operator[](const std::vector<int>& pos)
    {
        if (_modifications.empty())
            return at(pos);

        ///recovery initial positions for sub-tensor
        std::vector<int> rawPos;
        bool reducedDimenion = true;
        for (const auto& p : _modifications)
        {
            if (reducedDimenion)
                rawPos.push_back(p.first);
            else
                rawPos.back() += p.first;

            reducedDimenion = p.second == -1;
        }
        auto iteratorBegin = pos.cbegin();
        if (!reducedDimenion)
        {
            iteratorBegin++;
            rawPos.back() += pos.front();
        }
        rawPos.insert(rawPos.end(), iteratorBegin, pos.cend());

        return at(rawPos);
    }

    std::vector<int> dim() const
    {
        return _actuallyDimSizes;
    }

    Tensor operator()(int pos)
    {        
        const auto firtsDimSize = _actuallyDimSizes.front();
        if (pos >= firtsDimSize)
            throw std::invalid_argument("incorrect position");
 
        
        Tensor<float> tensor;
        tensor._data = _data;
        tensor._dimSizes = _dimSizes;

        tensor._actuallyDimSizes = std::vector<int>(_actuallyDimSizes.begin() + 1, _actuallyDimSizes.end()); 
        tensor._modifications = _modifications;
        tensor._modifications.push_back(std::make_pair(pos, -1));
        
        return tensor;
    }

    Tensor operator()(int low, int high)
    {
        if (low >= high)
            throw std::invalid_argument("incorrect low and high");

        const auto firtsDimSize = _actuallyDimSizes.front();
        if (low > firtsDimSize || high > firtsDimSize)
            throw std::invalid_argument("incorrect low and high");
        
        Tensor<float> tensor;
        tensor._data = _data;
        tensor._dimSizes = _dimSizes;

        tensor._actuallyDimSizes = _actuallyDimSizes;
        tensor._actuallyDimSizes[0] = high - low;
        tensor._modifications = _modifications;
        tensor._modifications.push_back(std::make_pair(low, high));

        return tensor;
    }

  
   Tensor reshape(const std::vector<int> &dimSizes)
    {
       //TODO implement for initialized _modifications
       if(!_modifications.empty())
            throw std::logic_error("sub-tensor does not support reshaping");
       
        Tensor<T> tensor;
        tensor._data = _data;
        tensor._dimSizes = dimSizes;
        tensor._actuallyDimSizes = _actuallyDimSizes;

        return tensor;
    }

private:
    std::shared_ptr<std::vector<T>> _data; ///< initial data
    std::vector<int> _dimSizes; ///< raw initial shape
    std::vector<int> _actuallyDimSizes; ///< actually shape after operator()
    std::vector<std::pair<int, int>> _modifications; ///< contain history of operator(), operator(int pos) - pair(pos, -1), operator(int low, int high) - pair(low, high)  


    /*! return element of initial data by positions */
    T& at(const std::vector<int>& pos)
    {
        size_t index = 0;
        size_t multiplier = 1;
        for (int i = pos.size() - 1; i >= 0; --i)
        {
            index += multiplier * pos[i];
            multiplier *= _dimSizes[i];
        }

        return _data->at(index);
    }
};


