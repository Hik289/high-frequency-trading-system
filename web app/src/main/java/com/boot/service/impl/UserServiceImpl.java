package com.boot.service.impl;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.boot.entity.User;
import com.boot.repository.UserRepository;
import com.boot.service.UserService;

@Service
public class UserServiceImpl implements UserService {
	@Autowired
	private UserRepository UserRepository;

	@Override
	public List<User> getList() {
		return UserRepository.findAll();
	}

}
