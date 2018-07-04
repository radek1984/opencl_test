/*
  main.cc

   Created on: Mar 26, 2018
       Author: radek
 */

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <assert.h>

#define SIZ (1 * 4096)
const char * helloStr  = R"(
__kernel void hello(
					const global float *x,
					global float *y
					)
{
	int gs = get_global_size(0);
	int ix = get_global_id(0);

}
)";

int main()
{
	cl_int err = CL_SUCCESS;
	try {
////////////PLATFORM INFO: /////////
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if (platforms.size() == 0) {
			std::cout << "Platform size 0\n";
			return -1;
		}

		for(const cl::Platform &p : platforms)
		{
			std::string s;
			p.getInfo(CL_PLATFORM_PROFILE, &s);
			std::cout<<"***********"<<std::endl;
			std::cout<<s<<std::endl;
			p.getInfo(CL_PLATFORM_VERSION, &s);
			std::cout<<s<<std::endl;
			p.getInfo(CL_PLATFORM_NAME, &s);
			std::cout<<s<<std::endl;
			p.getInfo(CL_PLATFORM_VENDOR, &s);
			std::cout<<s<<std::endl;
			p.getInfo(CL_PLATFORM_EXTENSIONS, &s);
			std::cout<<s<<std::endl<<std::endl;

			std::vector<cl::Device> devs;
			p.getDevices(CL_DEVICE_TYPE_ALL, &devs);
			std::cout<<"Devs count:" <<devs.size()<<std::endl;
			std::cout<<"DEVs:"<<std::endl<<std::endl;
			for(cl::Device &d : devs)
			{
				std::vector<size_t> sizes;
				d.getInfo(CL_DEVICE_NAME, &s);
				std::cout <<s<<std::endl;
				d.getInfo(CL_DEVICE_VENDOR, &s);
				std::cout <<s<<std::endl;
				d.getInfo(CL_DEVICE_PROFILE, &s);
				std::cout <<s<<std::endl;
				d.getInfo(CL_DEVICE_VERSION, &s);
				std::cout <<s<<std::endl;
				d.getInfo(CL_DRIVER_VERSION, &s);
				std::cout <<s<<std::endl;
				d.getInfo(CL_DEVICE_OPENCL_C_VERSION, &s);
				std::cout <<s<<std::endl;
				d.getInfo(CL_DEVICE_EXTENSIONS, &s);
				std::cout <<s<<std::endl;
				std::cout<<"CL_DEVICE_MAX_WORK_ITEM_SIZES ";
				d.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &sizes);
				for(int v:sizes)
					std::cout <<v<<" ";
				std::cout<<std::endl;


				cl_uint l;
				size_t l2;
				d.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &l2);
				std::cout<<"CL_DEVICE_MAX_WORK_GROUP_SIZE "<<l2<<std::endl;
				d.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &l);
				std::cout<<"CL_DEVICE_MAX_COMPUTE_UNITS "<<l<<std::endl;

			}
		}
		std::cout<<std::endl<<"***********"<<std::endl;

		cl_context_properties properties[] =
		   { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
		cl::Context context(CL_DEVICE_TYPE_GPU, properties);

		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

///////// COMPILATION: /////////////
		cl::Program::Sources source(1,
			std::make_pair(helloStr, strlen(helloStr)));
		cl::Program program_ = cl::Program(context, source);
		program_.build(devices);
		cl::Kernel kernel(program_, "hello", &err);

//////// BUFFERS: //////////
		float x_data[SIZ];
		for(int i = 0; i < SIZ; i++)
			x_data[i] = 1.0;
		cl::Buffer x(context,
				CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
		        sizeof(float) * SIZ,
				x_data,
		        NULL);

		float yy = 0.0;
		cl::Buffer y(context,
				CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
		        sizeof(float) * 1,
				&yy,
		        NULL);
		kernel.setArg(0, x);
		kernel.setArg(1, y);

//////// ENQUEUE KERNEL: /////////
		cl::Event event;
		cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
		queue.enqueueNDRangeKernel(
			kernel,
			cl::NullRange,
			cl::NDRange(SIZ),
			cl::NullRange,
			NULL,
			&event);
		event.wait();

////////// STATS: ///////////////
		cl_int res;
		cl_ulong tq, tsub, tstart, tend;
		res = event.getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, &tq);
		assert(res == CL_SUCCESS);
		res = event.getProfilingInfo(CL_PROFILING_COMMAND_SUBMIT, &tsub);
		assert(res == CL_SUCCESS);
		res = event.getProfilingInfo(CL_PROFILING_COMMAND_START, &tstart);
		assert(res == CL_SUCCESS);
		res = event.getProfilingInfo(CL_PROFILING_COMMAND_END, &tend);
		assert(res == CL_SUCCESS);

		std::cout<<"send: "<<(tsub - tq) / 1000<<"[us], setup: "<<(tstart - tsub) / 1000<<"[us], exec: "
				<<(tend - tstart) / 1000<<"[us]\n";

		queue.enqueueReadBuffer(y, CL_TRUE, 0, sizeof(float), &yy, NULL, NULL);
		std::cout<<"Res: "<<yy<<"\n";

	}
	catch (cl::Error &err) {
	 std::cerr
		<< "ERROR: "
		<< err.what()
		<< "("
		<< err.err()
		<< ")"
		<< std::endl;
	}

	return EXIT_SUCCESS;
}
