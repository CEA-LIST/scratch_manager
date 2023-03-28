Name:           scratch_manager
Version:        0.0.0
Release:        1%{?dist}
Summary:        A daemon to automate caching of read-only datasets.

License:        CeCILL-C
URL:            https://github.com/CEA-LIST/scratch_manager
Source0:        file://%{name}-%{version}.tar.gz

BuildArch:      noarch

BuildRequires:  git
BuildRequires:  python3-rpm-macros
BuildRequires:  python3-wheel
BuildRequires:  python3-setuptools
BuildRequires:  python3-setuptools_scm

Requires:       python3 >= 3.7

%description
Scratch manager is a daemon which automates caching for read only dataset on a system configuration with a slow and a fast storage device.
In a typical HPC environement, the slow storage is a shared network filesystem and the fast one is an ssd drive on the compute node.
It monitors read throughput on a list of dataset stored on a large but slow storage and moves the most active ones to a faster but limited cache storage.


%prep
%autosetup


%build
%{__python3} setup.py build


%install
%{__python3} setup.py install --prefix=%{_prefix} --root=%{buildroot}
install -o root -m 0644 -D scratch_manager.service %{buildroot}/%{_prefix}/lib/systemd/system/scratch_manager.service


%files
%license LICENSE.txt
%{_bindir}/%{name}
%{python3_sitelib}/%{name}/
%{python3_sitelib}/%{name}-%{version}*
%{_prefix}/lib/systemd/system/scratch_manager.service

%changelog