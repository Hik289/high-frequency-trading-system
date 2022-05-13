package com.boot.service.impl;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.boot.dao.FilesDao;
import com.boot.entity.Files;
import com.boot.repository.FilesRepository;
import com.boot.service.FilesService;

@Service
public class FilesServiceImpl implements FilesService {
	@Autowired
	private FilesRepository filesRepository;
	@Autowired
	private FilesDao filesDao;

	@Override
	public void save(Files f) {
		filesRepository.save(f);
	}

	@Override
	public List<Files> getListByCard(Integer cardId) {
		return filesDao.getListByCard(cardId);
	}

}
