/*
 * vector.cpp
 *
 *  Created on: 2019-04-28
 *      Author: 
 */

#include "headers/vector.h"
#include "omp.h"
#include <cassert>
#include <cmath>


Vector::Vector()
{}

Vector::Vector(std::size_t sz, double val)
: std::vector<double>(sz, val)
{
}

Vector::~Vector()
{}



Vector& Vector::operator=(double d)
{
	for(std::size_t i = 0; i < this->size(); ++i){
		(*this)[i] = d;
	}
	return *this;
}


Vector& Vector::operator+=(const Vector& v)
{
    assert(this->size() == v.size());
	for(std::size_t i = 0; i < this->size(); ++i){
		(*this)[i] += v[i];
	}
	return *this;
}

Vector& Vector::operator-=(const Vector& v)
{
    assert(this->size() == v.size());
	for(std::size_t i = 0; i < this->size(); ++i){
		(*this)[i] -= v[i];
	}
	return *this;
}


Vector& Vector::operator*=(double d)
{
	for(std::size_t i = 0; i < this->size(); ++i){
		(*this)[i] *= d;
	}
	return *this;
}


Vector Vector::operator+(const Vector& v) const
{
    assert(this->size() == v.size());
	Vector result(this->size());
	for(std::size_t i = 0; i < this->size(); ++i){
		result[i] = (*this)[i] + v[i];
	}
	return result;
}


Vector Vector::operator-(const Vector& v) const
{
    assert(this->size() == v.size());
	Vector result(this->size());
	for(std::size_t i = 0; i < this->size(); ++i){
		result[i] = (*this)[i] - v[i];
	}
	return result;
}


Vector Vector::operator*(double d) const
{
    Vector result(this->size());
	for(std::size_t i = 0; i < this->size(); ++i){
		result[i] = (*this)[i] * d;
	}
	return result;
}


double Vector::operator*(const Vector& v) const
{
    assert(this->size() == v.size());
	double result = 0.0;
	for(std::size_t i = 0; i < this->size(); ++i){
		result += (*this)[i] * v[i];
	}
	return result;
}


double Vector::norm()
{
    double sum = 0.0;
	for(std::size_t i = 0; i < this->size(); ++i){
		sum += (*this)[i] * (*this)[i];
	}
    return std::sqrt(sum);
}

double Vector::norm() const
{
	double sum = 0.0;
	for(std::size_t i = 0; i < this->size(); ++i){
		sum += (*this)[i] * (*this)[i];
	}
	return std::sqrt(sum);
}


std::ostream& operator<<(std::ostream& stream, const Vector& v)
{
	if (v.empty()) return stream << "()";

	std::size_t sz = v.size() - 1;
	stream << "(";
	for (std::size_t i = 0; i < sz; ++i)
		stream << v[i] << " ";
	stream << v[sz] << ")";

	return stream;
}

